#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

try:
    from ..ir import *
    from ..dialects import transform
except ImportError as e:
    raise RuntimeError("Error loading imports from extension module") from e

from typing import Optional, Sequence, Union, overload


class MapForallToBlocks:
    """Specialization for MapForallToBlocks class."""

    @overload
    def __init__(
        self,
        result_type: Type,
        target: Union[Operation, OpView, Value],
        *,
        grid_dims: Optional[Sequence[int]] = None,
        generate_gpu_launch: Optional[bool] = None,
        loc=None,
        ip=None
    ):
        ...

    @overload
    def __init__(
        self,
        target: Union[Operation, OpView, Value],
        *,
        grid_dims: Optional[Sequence[int]] = None,
        generate_gpu_launch: Optional[bool] = None,
        loc=None,
        ip=None
    ):
        ...

    def __init__(
        self,
        result_type_or_target: Union[Operation, OpView, Type, Value],
        target_or_none: Optional[Union[Operation, OpView, Value]] = None,
        *,
        grid_dims: Optional[Sequence[int]] = None,
        generate_gpu_launch: Optional[bool] = None,
        loc=None,
        ip=None
    ):
        if isinstance(result_type_or_target, Type):
            result_type = result_type_or_target
            target = target_or_none
        else:
            result_type = transform.AnyOpType.get()
            target = result_type_or_target

        if grid_dims is not None and not isinstance(grid_dims, ArrayAttr):
            grid_dims = DenseI64ArrayAttr.get(grid_dims)

        super().__init__(
            result_type,
            target,
            grid_dims=grid_dims,
            generate_gpu_launch=generate_gpu_launch,
            loc=loc,
            ip=ip,
        )
