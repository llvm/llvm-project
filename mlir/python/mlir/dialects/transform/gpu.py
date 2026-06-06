#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from .._gpu_transform_ops_gen import *
from .._gpu_transform_ops_gen import _Dialect

try:
    from ...ir import *
    from ...dialects import transform
    from .._ods_common import _cext as _ods_cext
except ImportError as e:
    raise RuntimeError("Error loading imports from extension module") from e

from typing import Optional, Sequence, Union, overload


@_ods_cext.register_operation(_Dialect, replace=True)
class MapForallToBlocks(MapForallToBlocks):
    """Specialization for MapForallToBlocks class."""

    @overload
    def __init__(
        self,
        result_type: Type,
        target: Union[Operation, OpView, Value],
        *,
        grid_dims: Optional[Union[Sequence[int], Attribute]] = None,
        generate_gpu_launch: Optional[Union[bool, Attribute]] = None,
        loc=None,
        ip=None,
    ):
        ...

    @overload
    def __init__(
        self,
        target: Union[Operation, OpView, Value],
        *,
        grid_dims: Optional[Union[Sequence[int], Attribute]] = None,
        generate_gpu_launch: Optional[Union[bool, Attribute]] = None,
        loc=None,
        ip=None,
    ):
        ...

    def __init__(
        self,
        result_type_or_target: Union[Operation, OpView, Type, Value],
        target_or_none: Optional[Union[Operation, OpView, Value]] = None,
        *,
        grid_dims: Optional[Union[Sequence[int], Attribute]] = None,
        generate_gpu_launch: Optional[Union[bool, Attribute]] = None,
        loc=None,
        ip=None,
    ):
        if isinstance(result_type_or_target, Type):
            result_type = result_type_or_target
            target = target_or_none
        else:
            result_type = transform.AnyOpType.get()
            target = result_type_or_target

        super().__init__(
            result_type,
            target,
            grid_dims=grid_dims,
            generate_gpu_launch=generate_gpu_launch,
            loc=loc,
            ip=ip,
        )


@_ods_cext.register_operation(_Dialect, replace=True)
class MapNestedForallToThreads(MapNestedForallToThreads):
    """Specialization for MapNestedForallToThreads class."""

    @overload
    def __init__(
        self,
        result_type: Type,
        target: Union[Operation, OpView, Value],
        *,
        block_dims: Optional[Sequence[int]] = None,
        warp_size: Optional[Sequence[int]] = None,
        sync_after_distribute: Optional[bool] = None,
        loc=None,
        ip=None,
    ):
        ...

    @overload
    def __init__(
        self,
        target: Union[Operation, OpView, Value],
        *,
        block_dims: Optional[Sequence[int]] = None,
        warp_size: Optional[Sequence[int]] = None,
        sync_after_distribute: Optional[bool] = None,
        loc=None,
        ip=None,
    ):
        ...

    def __init__(
        self,
        result_type_or_target: Union[Operation, OpView, Value, Type],
        target_or_none: Optional[Union[Operation, OpView, Value]] = None,
        *,
        block_dims: Optional[Union[Sequence[int], Attribute]] = None,
        warp_size: Optional[Union[Sequence[int], Attribute]] = None,
        sync_after_distribute: Optional[bool] = None,
        loc=None,
        ip=None,
    ):
        if isinstance(result_type_or_target, Type):
            result_type = result_type_or_target
            target = target_or_none
        else:
            result_type = result_type_or_target.type
            target = result_type_or_target
        super().__init__(
            result_type,
            target,
            block_dims=block_dims,
            warp_size=warp_size,
            sync_after_distribute=sync_after_distribute,
            loc=loc,
            ip=ip,
        )
