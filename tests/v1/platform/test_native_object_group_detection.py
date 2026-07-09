# SPDX-License-Identifier: Apache-2.0

# Standard
import types

# Third Party
import pytest

# First Party
from lmcache.v1.platform import _torch_ops
from lmcache.v1.platform.base_device_ops import OPS, DeviceOps


def test_baseline_object_group_transfer_detection_is_false() -> None:
    """Torch-baseline execute_object_group_transfer is detected as non-native."""
    target = types.ModuleType("baseline_ops_probe")
    DeviceOps.populate_module(target)

    assert (
        target.execute_object_group_transfer
        is not _torch_ops.execute_object_group_transfer
    ) is False


def test_bound_native_object_group_transfer_detection_is_true() -> None:
    """Binding native execute_object_group_transfer flips native detection."""

    class _FakeNative:
        @staticmethod
        def execute_object_group_transfer(*_args: object, **_kwargs: object) -> None:
            return None

    target = types.ModuleType("native_ops_probe")
    DeviceOps.populate_module(target)
    DeviceOps._bind_native(target, _FakeNative())

    assert (
        target.execute_object_group_transfer
        is not _torch_ops.execute_object_group_transfer
    )


def test_execute_object_group_transfer_stays_in_ops_contract() -> None:
    """Object-group transfer remains part of the DeviceOps contract."""
    assert "execute_object_group_transfer" in OPS


def test_torch_baseline_execute_object_group_transfer_raises() -> None:
    """Baseline object-group transfer stub always raises NotImplementedError."""
    with pytest.raises(NotImplementedError):
        _torch_ops.execute_object_group_transfer(
            _torch_ops.TransferDirection.D2H,
            "cpu",
            64,
            [],
            [],
        )
