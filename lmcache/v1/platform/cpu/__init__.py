# SPDX-License-Identifier: Apache-2.0
"""CPU-specific platform primitives.

:class:`~lmcache.v1.platform.cpu.shm.CpuShmTensorWrapper` carries a
``device_type`` ClassVar and a ``wrap`` factory classmethod, which
:func:`~lmcache.v1.platform._registry._discover_wrappers_once` picks
up at run-time -- no static ``register_kv_wrapper`` needed.
"""

# First Party
from lmcache.v1.platform.base_device_spec import DeviceSpec


class CpuDeviceSpec(DeviceSpec):
    """CPU device specification for the detection registry."""

    @property
    def device_type(self) -> str:
        return "cpu"

    @property
    def torch_module_name(self) -> str:
        return "cpu"

    @property
    def ops_cls(self) -> "type[DeviceOps]":
        # First Party
        from lmcache.v1.platform.cpu.device_ops import CpuDeviceOps

        return CpuDeviceOps
