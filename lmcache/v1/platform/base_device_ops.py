# SPDX-License-Identifier: Apache-2.0
"""The unified per-device ops abstraction (the ``lmcache.c_ops`` surface).

:class:`DeviceOps` owns the full op contract (:data:`OPS`) with a
device-agnostic torch baseline migrated into
:mod:`lmcache.v1.platform._torch_ops`. The base class *is* the CPU/torch
backend: :meth:`DeviceOps.populate_module` installs each op as a direct
reference to the corresponding ``_torch_ops`` function. Accelerator subclasses
override :meth:`populate_module` to layer native ops on top of the baseline.

The base itself has an empty ``device_type`` so it is never registered as a
device -- it is pure baseline + contract. Concrete devices live in
``platform/<backend>/device_ops.py`` and are discovered by ``device_type``.
"""

# Future
from __future__ import annotations

# Standard
from typing import ClassVar

# First Party
from lmcache.v1.platform import _torch_ops
from lmcache.v1.platform.ops_types import (
    BatchStep,
    EngineKVFormat,
    KernelGroupSpec,
    LaunchVar,
    PageBufferShapeDesc,
    StagingCopy,
    TransferDirection,
    set_shape_desc_dtype,
)

#: The complete ops contract: every name a device provides natively or inherits
#: from the torch baseline. Single source of truth for the parity/contract test,
#: :meth:`DeviceOps.populate_module`, and the ``lmcache.c_ops`` shim.
OPS: frozenset[str] = frozenset(
    {
        "alloc_pinned_numa_ptr",
        "free_pinned_numa_ptr",
        "alloc_pinned_ptr",
        "free_pinned_ptr",
        "alloc_shm_pinned_ptr",
        "free_shm_pinned_ptr",
        "alloc_hugepage_pinned_ptr",
        "free_hugepage_pinned_ptr",
        "alloc_hugepage_pinned_numa_ptr",
        "free_hugepage_pinned_numa_ptr",
        "alloc_numa_ptr",
        "free_numa_ptr",
        "batched_memcpy",
        "lmcache_memcpy_async",
        "multi_layer_kv_transfer",
        "multi_layer_kv_transfer_unilateral",
        "multi_layer_block_kv_transfer",
        "single_layer_kv_transfer",
        "single_layer_kv_transfer_sgl",
        "load_and_reshape_flash",
        "reshape_and_cache_back_flash",
        "encode_fast_new",
        "decode_fast_new",
        "decode_fast_prefsum",
        "calculate_cdf",
        "rotary_embedding_k_fused",
        "get_gpu_pci_bus_id",
        "record_completion_on_stream",
        "drain_recorded_completions",
        "record_event_on_stream",
        "drain_recorded_events",
        "is_cross_layer",
        "is_kv_list",
        "is_layer_list",
        "is_mla",
        "execute_object_group_transfer",
    }
)

#: Shared type names that a native pybind module may export and that should be
#: rebound onto the shim so identity/``isinstance`` checks line up with the
#: native structs. ``GPUKVFormat`` is a back-compat alias of ``EngineKVFormat``
#: and handled specially in :meth:`DeviceOps._bind_native_types`.
NATIVE_TYPE_NAMES: tuple[str, ...] = (
    "TransferDirection",
    "EngineKVFormat",
    "PageBufferShapeDesc",
    "StagingCopy",
    "LaunchVar",
    "BatchStep",
    "KernelGroupSpec",
)


class DeviceOps:
    """Strategy base: classmethod-driven ops population for one device type.

    Concrete subclasses set :attr:`device_type` and override
    :meth:`populate_module` to layer native ops on top of the torch baseline.
    The base itself has no ``device_type`` (empty), so it is never registered
    as a device.

    Call ``DeviceOps.populate_module(target)`` (or a subclass) to install all
    ops + shared types onto *target* (typically a :class:`types.ModuleType`).
    """

    device_type: ClassVar[str] = ""  # base is unregistered

    # Shared types as class attributes (for direct access and test assertions).
    TransferDirection = TransferDirection
    EngineKVFormat = EngineKVFormat
    GPUKVFormat = EngineKVFormat  # back-compat alias
    PageBufferShapeDesc = PageBufferShapeDesc
    set_shape_desc_dtype = staticmethod(set_shape_desc_dtype)

    # Object-group transfer plan types (native-only; stubs on the baseline).
    StagingCopy = StagingCopy
    LaunchVar = LaunchVar
    BatchStep = BatchStep
    KernelGroupSpec = KernelGroupSpec

    @classmethod
    def populate_module(cls, target: object) -> None:
        """Install all ops and shared types onto *target*.

        The base implementation installs the torch baseline for every op in
        :data:`OPS`. Subclasses override to layer native ops on top::

            @classmethod
            def populate_module(cls, target):
                super().populate_module(target)  # torch baseline first
                import my_native_module as native
                cls._bind_native(target, native)
                cls._bind_native_types(target, native)  # rebind native types

        Args:
            target: The object to set attributes on (typically a module or
                namespace used as the ``lmcache.c_ops`` shim).
        """
        # Install torch baseline ops as direct function references.
        for name in OPS:
            setattr(target, name, getattr(_torch_ops, name))
        # Install shared types.
        target.TransferDirection = cls.TransferDirection  # type: ignore[attr-defined]
        target.EngineKVFormat = cls.EngineKVFormat  # type: ignore[attr-defined]
        target.GPUKVFormat = cls.GPUKVFormat  # type: ignore[attr-defined]
        target.PageBufferShapeDesc = cls.PageBufferShapeDesc  # type: ignore[attr-defined]
        target.set_shape_desc_dtype = cls.set_shape_desc_dtype  # type: ignore[attr-defined]
        target.StagingCopy = cls.StagingCopy  # type: ignore[attr-defined]
        target.LaunchVar = cls.LaunchVar  # type: ignore[attr-defined]
        target.BatchStep = cls.BatchStep  # type: ignore[attr-defined]
        target.KernelGroupSpec = cls.KernelGroupSpec  # type: ignore[attr-defined]

    @classmethod
    def _bind_native(cls, target: object, module: object) -> None:
        """Layer a compiled module's native ops onto *target*.

        Iterates :data:`OPS` and, for each name the module exports, overwrites
        the attribute on *target* with the native callable. Names absent from
        the module keep whatever was previously installed (the torch baseline).
        """
        for name in OPS:
            fn = getattr(module, name, None)
            if fn is not None:
                setattr(target, name, fn)

    @classmethod
    def _bind_native_types(cls, target: object, module: object) -> None:
        """Rebind shared pybind *types* from a compiled module onto *target*.

        Restores the behavior of the former ``__dict__.update`` merge: whatever
        shared types the native module exports replace the pure-Python
        ``ops_types`` stubs, so identity / ``isinstance`` checks (and passing a
        value into a native function that expects the pybind struct) line up
        with the native boundary. Enums already compare by value via
        :class:`~enum.IntEnum`; this fixes the *identity* side.

        Types the module does not export are skipped, so this is safe for
        partial backends: the SYCL ``xpu_ops`` module, for example, only exports
        ``TransferDirection`` / ``EngineKVFormat`` (+ the ``GPUKVFormat`` alias)
        and no ``PageBufferShapeDesc`` / plan structs, and those simply keep the
        Python stubs already installed by :meth:`populate_module`.
        """
        for name in NATIVE_TYPE_NAMES:
            native_type = getattr(module, name, None)
            if native_type is not None:
                setattr(target, name, native_type)
        # ``GPUKVFormat`` is a back-compat alias of ``EngineKVFormat``. Prefer a
        # native alias if the module exports one; otherwise mirror whatever
        # ``EngineKVFormat`` resolved to above so the alias never diverges.
        native_alias = getattr(module, "GPUKVFormat", None)
        if native_alias is not None:
            target.GPUKVFormat = native_alias  # type: ignore[attr-defined]
        elif getattr(module, "EngineKVFormat", None) is not None:
            target.GPUKVFormat = module.EngineKVFormat  # type: ignore[attr-defined]
