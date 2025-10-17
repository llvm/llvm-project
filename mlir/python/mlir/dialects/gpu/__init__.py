#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from .._gpu_ops_gen import *
from .._gpu_ops_gen import _Dialect
from .._gpu_enum_gen import *
from ..._mlir_libs._mlirDialectsGPU import *
from typing import Callable, Sequence, Union, Optional, List

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
    )
    from .._ods_common import (
        get_default_loc_context as _get_default_loc_context,
        _cext as _ods_cext,
    )
except ImportError as e:
    raise RuntimeError("Error loading imports from extension module") from e


@_ods_cext.register_operation(_Dialect, replace=True)
class GPUFuncOp(GPUFuncOp):
    __doc__ = GPUFuncOp.__doc__

    KERNEL_ATTR_NAME = "gpu.kernel"
    KNOWN_BLOCK_SIZE_ATTR_NAME = "known_block_size"
    KNOWN_GRID_SIZE_ATTR_NAME = "known_grid_size"

    FUNCTION_TYPE_ATTR_NAME = "function_type"
    SYM_NAME_ATTR_NAME = "sym_name"
    ARGUMENT_ATTR_NAME = "arg_attrs"
    RESULT_ATTR_NAME = "res_attrs"

    def __init__(
        self,
        function_type: Union[FunctionType, TypeAttr],
        sym_name: Optional[Union[str, StringAttr]] = None,
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
