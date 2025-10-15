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


KERNEL_ATTRIBUTE_NAME = "gpu.kernel"
KNOWN_BLOCK_SIZE_ATTRIBUTE_NAME = "gpu.known_block_size"
KNOWN_GRID_SIZE_ATTRIBUTE_NAME = "gpu.known_grid_size"

FUNCTION_TYPE_ATTRIBUTE_NAME = "function_type"
SYM_NAME_ATTRIBUTE_NAME = "sym_name"
ARGUMENT_ATTRIBUTE_NAME = "arg_attrs"
RESULT_ATTRIBUTE_NAME = "res_attrs"

@_ods_cext.register_operation(_Dialect, replace=True)
class GPUFuncOp(GPUFuncOp):
    __doc__ = GPUFuncOp.__doc__

    def __init__(
        self,
        function_type: Union[FunctionType, TypeAttr],
        sym_name: Optional[Union[str, StringAttr]] = None,
        kernel: Optional[bool] = None,
        body_builder: Optional[Callable[[GPUFuncOp], None]] = None,
        known_block_size: Optional[Union[List[int], DenseI32ArrayAttr]] = None,
        known_grid_size: Optional[Union[List[int], DenseI32ArrayAttr]] = None,
        *args,
        loc=None,
        ip=None,
        **kwargs,
    ):
        """
        Create a GPUFuncOp with the provided `function_type`, `sym_name`, `kernel`, `body_builder`, `known_block_size`, and `known_grid_size`.
        - `function_type` is a FunctionType or a TypeAttr.
        - `sym_name` is a string or a StringAttr representing the function name.
        - `kernel` is a boolean representing whether the function is a kernel.
        - `body_builder` is an optional callback. When provided, a new entry block
          is created and the callback is invoked with the new op as argument within
          an InsertionPoint context already set for the block. The callback is
          expected to insert a terminator in the block.
        - `known_block_size` is an optional list of integers or a DenseI32ArrayAttr representing the known block size.
        - `known_grid_size` is an optional list of integers or a DenseI32ArrayAttr representing the known grid size.
        """
        function_type = (
            TypeAttr.get(function_type)
            if not isinstance(function_type, TypeAttr)
            else function_type
        )
        super().__init__(function_type, *args, loc=loc, ip=ip, **kwargs)
        if sym_name is not None:
            self.attributes[SYM_NAME_ATTRIBUTE_NAME] = StringAttr.get(str(sym_name))
        if kernel:
            self.attributes[KERNEL_ATTRIBUTE_NAME] = UnitAttr.get()
        if body_builder is not None:
            with InsertionPoint(self.add_entry_block()):
                body_builder(self)
        if known_block_size is not None:
            if isinstance(known_block_size, list):
                self.attributes[KNOWN_BLOCK_SIZE_ATTRIBUTE_NAME] = (
                    DenseI32ArrayAttr.get(known_block_size)
                )
            else:
                self.attributes[KNOWN_BLOCK_SIZE_ATTRIBUTE_NAME] = known_block_size
        if known_grid_size is not None:
            if isinstance(known_grid_size, list):
                self.attributes[KNOWN_GRID_SIZE_ATTRIBUTE_NAME] = (
                    DenseI32ArrayAttr.get(known_grid_size)
                )
            else:
                self.attributes[KNOWN_GRID_SIZE_ATTRIBUTE_NAME] = known_grid_size

    @property
    def type(self) -> FunctionType:
        return FunctionType(
            TypeAttr(self.attributes[FUNCTION_TYPE_ATTRIBUTE_NAME]).value
        )

    @property
    def name(self) -> StringAttr:
        return StringAttr(self.attributes[SYM_NAME_ATTRIBUTE_NAME])

    @property
    def is_kernel(self) -> bool:
        return KERNEL_ATTRIBUTE_NAME in self.attributes

    @property
    def known_block_size(self) -> Optional[DenseI32ArrayAttr]:
        if KNOWN_BLOCK_SIZE_ATTRIBUTE_NAME not in self.attributes:
            return None
        return self.attributes[KNOWN_BLOCK_SIZE_ATTRIBUTE_NAME]

    @property
    def known_grid_size(self) -> Optional[DenseI32ArrayAttr]:
        if KNOWN_GRID_SIZE_ATTRIBUTE_NAME not in self.attributes:
            return None
        return self.attributes[KNOWN_GRID_SIZE_ATTRIBUTE_NAME]

    def add_entry_block(self) -> Block:
        function_type = self.type
        return self.body.blocks.append(
            *function_type.inputs,
            arg_locs=[self.location for _ in function_type.inputs],
        )

    @property
    def entry_block(self) -> Block:
        return self.body.blocks[0]

    @property
    def arguments(self) -> Sequence[Type]:
        return self.type.inputs

    @property
    def arg_attrs(self):
        if ARGUMENT_ATTRIBUTE_NAME not in self.attributes:
            return ArrayAttr.get([DictAttr.get({}) for _ in self.type.inputs])
        return ArrayAttr(self.attributes[ARGUMENT_ATTRIBUTE_NAME])

    @arg_attrs.setter
    def arg_attrs(self, attribute: Union[ArrayAttr, list[Attribute]]):
        if isinstance(attribute, ArrayAttr):
            self.attributes[ARGUMENT_ATTRIBUTE_NAME] = attribute
        else:
            self.attributes[ARGUMENT_ATTRIBUTE_NAME] = ArrayAttr.get(
                attribute, context=self.context
            )

    @property
    def result_attrs(self) -> Optional[ArrayAttr]:
        if RESULT_ATTRIBUTE_NAME not in self.attributes:
            return ArrayAttr.get([DictAttr.get({}) for _ in self.type.results])
        return self.attributes[RESULT_ATTRIBUTE_NAME]

    @result_attrs.setter
    def result_attrs(self, attribute: Union[ArrayAttr, list[Attribute]]):
        if isinstance(attribute, ArrayAttr):
            self.attributes[RESULT_ATTRIBUTE_NAME] = attribute
        else:
            self.attributes[RESULT_ATTRIBUTE_NAME] = ArrayAttr.get(
                attribute, context=self.context
            )
