#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from .._gpu_ops_gen import *
from .._gpu_ops_gen import _Dialect
from .._gpu_enum_gen import *
from ..._mlir_libs._mlirDialectsGPU import *
from typing import Any, Callable, Sequence, Tuple, Union, Optional, List

try:
    from ...ir import (
        FunctionType,
        TypeAttr,
        StringAttr,
        UnitAttr,
        Block,
        InsertionPoint,
        ArrayAttr,
        Type,
        DictAttr,
        Attribute,
        DenseI32ArrayAttr,
        Value,
    )
    from ...extras.meta import region_op
    from ...extras import types as T
    from ..arith import constant, ConstantOp
    from .._ods_common import (
        get_default_loc_context as _get_default_loc_context,
        _cext as _ods_cext,
        get_op_result_or_op_results,
    )
except ImportError as e:
    raise RuntimeError("Error loading imports from extension module") from e


def gpu_async_token():
    return Type.parse("!gpu.async.token")


@_ods_cext.register_operation(_Dialect, replace=True)
class GPUFuncOp(GPUFuncOp):
    __doc__ = GPUFuncOp.__doc__

    KERNEL_ATTR_NAME = "gpu.kernel"
    KNOWN_BLOCK_SIZE_ATTR_NAME = "known_block_size"
    KNOWN_GRID_SIZE_ATTR_NAME = "known_grid_size"

    FUNCTION_TYPE_ATTR_NAME = "function_type"
    SYM_NAME_ATTR_NAME = "sym_name"

    def __init__(
        self,
        function_type: Union[FunctionType, TypeAttr],
        sym_name: Optional[Union[str, StringAttr]] = None,
        arg_attrs: Optional[Sequence[dict]] = None,
        res_attrs: Optional[Sequence[dict]] = None,
        kernel: Optional[bool] = None,
        workgroup_attrib_attrs: Optional[Sequence[dict]] = None,
        private_attrib_attrs: Optional[Sequence[dict]] = None,
        known_block_size: Optional[Union[Sequence[int], DenseI32ArrayAttr]] = None,
        known_grid_size: Optional[Union[Sequence[int], DenseI32ArrayAttr]] = None,
        loc=None,
        ip=None,
        body_builder: Optional[Callable[[GPUFuncOp], None]] = None,
    ):
        """
        Create a GPUFuncOp with the provided `function_type`, `sym_name`,
        `kernel`, `workgroup_attrib_attrs`, `private_attrib_attrs`, `known_block_size`,
        `known_grid_size`, and `body_builder`.
        - `function_type` is a FunctionType or a TypeAttr.
        - `sym_name` is a string or a StringAttr representing the function name.
        - `kernel` is a boolean representing whether the function is a kernel.
        - `workgroup_attrib_attrs` is an optional list of dictionaries.
        - `private_attrib_attrs` is an optional list of dictionaries.
        - `known_block_size` is an optional list of integers or a DenseI32ArrayAttr representing the known block size.
        - `known_grid_size` is an optional list of integers or a DenseI32ArrayAttr representing the known grid size.
        - `body_builder` is an optional callback. When provided, a new entry block
          is created and the callback is invoked with the new op as argument within
          an InsertionPoint context already set for the block. The callback is
          expected to insert a terminator in the block.
        """
        function_type = (
            TypeAttr.get(function_type)
            if not isinstance(function_type, TypeAttr)
            else function_type
        )
        super().__init__(
            function_type,
            arg_attrs=arg_attrs,
            res_attrs=res_attrs,
            workgroup_attrib_attrs=workgroup_attrib_attrs,
            private_attrib_attrs=private_attrib_attrs,
            loc=loc,
            ip=ip,
        )

        if isinstance(sym_name, str):
            self.attributes[self.SYM_NAME_ATTR_NAME] = StringAttr.get(sym_name)
        elif isinstance(sym_name, StringAttr):
            self.attributes[self.SYM_NAME_ATTR_NAME] = sym_name
        else:
            raise ValueError("sym_name must be a string or a StringAttr")

        if kernel:
            self.attributes[self.KERNEL_ATTR_NAME] = UnitAttr.get()

        if known_block_size is not None:
            if isinstance(known_block_size, Sequence):
                block_size = DenseI32ArrayAttr.get(known_block_size)
                self.attributes[self.KNOWN_BLOCK_SIZE_ATTR_NAME] = block_size
            elif isinstance(known_block_size, DenseI32ArrayAttr):
                self.attributes[self.KNOWN_BLOCK_SIZE_ATTR_NAME] = known_block_size
            else:
                raise ValueError(
                    "known_block_size must be a list of integers or a DenseI32ArrayAttr"
                )

        if known_grid_size is not None:
            if isinstance(known_grid_size, Sequence):
                grid_size = DenseI32ArrayAttr.get(known_grid_size)
                self.attributes[self.KNOWN_GRID_SIZE_ATTR_NAME] = grid_size
            elif isinstance(known_grid_size, DenseI32ArrayAttr):
                self.attributes[self.KNOWN_GRID_SIZE_ATTR_NAME] = known_grid_size
            else:
                raise ValueError(
                    "known_grid_size must be a list of integers or a DenseI32ArrayAttr"
                )

        if body_builder is not None:
            with InsertionPoint(self.add_entry_block()):
                body_builder(self)

    @property
    def name(self) -> StringAttr:
        return StringAttr(self.attributes[self.SYM_NAME_ATTR_NAME])

    @property
    def is_kernel(self) -> bool:
        return self.KERNEL_ATTR_NAME in self.attributes

    def add_entry_block(self) -> Block:
        if len(self.body.blocks) > 0:
            raise RuntimeError(f"Entry block already exists for {self.name.value}")

        function_type = self.function_type.value
        return self.body.blocks.append(
            *function_type.inputs,
            arg_locs=[self.location for _ in function_type.inputs],
        )

    @property
    def entry_block(self) -> Block:
        if len(self.body.blocks) == 0:
            raise RuntimeError(
                f"Entry block does not exist for {self.name.value}."
                + " Do you need to call the add_entry_block() method on this GPUFuncOp?"
            )
        return self.body.blocks[0]

    @property
    def arguments(self) -> Sequence[Type]:
        return self.function_type.value.inputs


def _convert_literal_to_constant(value: Union[int, ConstantOp, Value]) -> Value:
    if isinstance(value, int):
        return constant(T.index(), value)
    elif isinstance(value, (ConstantOp, Value)):
        return value
    else:
        raise ValueError(f"Invalid value: {value}")


@_ods_cext.register_operation(_Dialect, replace=True)
class LaunchFuncOp(LaunchFuncOp):
    __doc__ = LaunchFuncOp.__doc__

    def __init__(
        self,
        kernel: List[str],
        grid_size: Tuple[Any, Any, Any],
        block_size: Tuple[Any, Any, Any],
        kernel_operands: Optional[List[Value]] = None,
        async_dependencies: Optional[List[Value]] = None,
        dynamic_shared_memory_size: Optional[Value] = None,
        async_object=None,
        *,
        loc=None,
        ip=None,
    ):
        if async_dependencies is None:
            async_dependencies = []
        async_token = None
        if len(async_dependencies):
            async_token = gpu_async_token()

        grid_size_x, grid_size_y, grid_size_z = map(
            _convert_literal_to_constant, grid_size
        )
        block_size_x, block_size_y, block_size_z = map(
            _convert_literal_to_constant, block_size
        )

        super().__init__(
            async_token,
            async_dependencies,
            kernel,
            grid_size_x,
            grid_size_y,
            grid_size_z,
            block_size_x,
            block_size_y,
            block_size_z,
            kernel_operands,
            dynamicSharedMemorySize=dynamic_shared_memory_size,
            asyncObject=async_object,
            loc=loc,
            ip=ip,
        )


def launch_func(
    kernel: List[str],
    grid_size: Tuple[Any, Any, Any],
    block_size: Tuple[Any, Any, Any],
    kernel_operands: Optional[List[Value]] = None,
    async_dependencies: Optional[List[Value]] = None,
    dynamic_shared_memory_size: Optional[Value] = None,
    async_object=None,
    *,
    loc=None,
    ip=None,
) -> Union[Value, List[Value], LaunchFuncOp]:
    op = LaunchFuncOp(
        kernel=kernel,
        grid_size=grid_size,
        block_size=block_size,
        kernel_operands=kernel_operands,
        async_dependencies=async_dependencies,
        dynamic_shared_memory_size=dynamic_shared_memory_size,
        async_object=async_object,
        loc=loc,
        ip=ip,
    )
    results = op.results
    if len(results) == 1:
        return results[0]
    elif len(results) > 1:
        return results
    else:
        return op


def wait(
    async_dependencies: Optional[List[Value]] = None, *, loc=None, ip=None
) -> Union[Value, List[Value], WaitOp]:
    if async_dependencies is None:
        async_dependencies = []
    return get_op_result_or_op_results(
        WaitOp(gpu_async_token(), async_dependencies, loc=loc, ip=ip)
    )


@_ods_cext.register_operation(_Dialect, replace=True)
class LaunchOp(LaunchOp):
    __doc__ = LaunchOp.__doc__

    def __init__(
        self,
        grid_size: Tuple[Any, Any, Any],
        block_size: Tuple[Any, Any, Any],
        async_dependencies=None,
        dynamic_shared_memory_size: Optional[Value] = None,
        *,
        loc=None,
        ip=None,
    ):
        if async_dependencies is None:
            async_dependencies = []
        async_token = None
        if len(async_dependencies):
            async_token = gpu_async_token()
        grid_size_x, grid_size_y, grid_size_z = map(
            _convert_literal_to_constant, grid_size
        )
        block_size_x, block_size_y, block_size_z = map(
            _convert_literal_to_constant, block_size
        )

        super().__init__(
            async_token,
            async_dependencies,
            grid_size_x,
            grid_size_y,
            grid_size_z,
            block_size_x,
            block_size_y,
            block_size_z,
            dynamicSharedMemorySize=dynamic_shared_memory_size,
            loc=loc,
            ip=ip,
        )
        self.regions[0].blocks.append(*[T.index() for _ in range(12)])


def launch_(
    grid_size: Tuple[Any, Any, Any],
    block_size: Tuple[Any, Any, Any],
    async_dependencies=None,
    dynamic_shared_memory_size: Optional[Value] = None,
    *,
    loc=None,
    ip=None,
):
    grid_size = tuple(map(_convert_literal_to_constant, grid_size))
    block_size = tuple(map(_convert_literal_to_constant, block_size))
    launch_op = LaunchOp(
        grid_size,
        block_size,
        async_dependencies,
        dynamic_shared_memory_size,
        loc=loc,
        ip=ip,
    )
    return launch_op


launch = region_op(launch_, terminator=lambda *_args: terminator())


_printf = printf


def printf(format, *args, loc=None, ip=None):
    return _printf(format=format, args=args, loc=loc, ip=ip)
