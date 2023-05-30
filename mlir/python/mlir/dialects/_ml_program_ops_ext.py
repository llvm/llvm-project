#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

try:
    from typing import Union
    from ..ir import *
    from ._ods_common import get_default_loc_context as _get_default_loc_context
except ImportError as e:
    raise RuntimeError("Error loading imports from extension module") from e

from ._ml_program_ops_gen import *


ARGUMENT_ATTRIBUTE_NAME = "arg_attrs"
RESULT_ATTRIBUTE_NAME = "res_attrs"


class FuncOp:
    """Specialization for the func op class."""

    def __init__(
        self, name, type, *, visibility=None, body_builder=None, loc=None, ip=None
    ):
        """
        Create a FuncOp with the provided `name`, `type`, and `visibility`.
        - `name` is a string representing the function name.
        - `type` is either a FunctionType or a pair of list describing inputs and
          results.
        - `visibility` is a string matching `public`, `private`, or `nested`. None
          implies private visibility.
        - `body_builder` is an optional callback, when provided a new entry block
          is created and the callback is invoked with the new op as argument within
          an InsertionPoint context already set for the block. The callback is
          expected to insert a terminator in the block.
        """
        sym_name = StringAttr.get(str(name))

        # If the type is passed as a tuple, build a FunctionType on the fly.
        if isinstance(type, tuple):
            type = FunctionType.get(inputs=type[0], results=type[1])

        type = TypeAttr.get(type)
        sym_visibility = (
            StringAttr.get(str(visibility)) if visibility is not None else None
        )
        super().__init__(sym_name, type, sym_visibility=sym_visibility, loc=loc, ip=ip)
        if body_builder:
            entry_block = self.add_entry_block()
            with InsertionPoint(entry_block):
                body_builder(self)

    @property
    def is_external(self):
        return len(self.regions[0].blocks) == 0

    @property
    def body(self):
        return self.regions[0]

    @property
    def type(self):
        return FunctionType(TypeAttr(self.attributes["function_type"]).value)

    @property
    def visibility(self):
        return self.attributes["sym_visibility"]

    @property
    def name(self) -> StringAttr:
        return StringAttr(self.attributes["sym_name"])

    @property
    def entry_block(self):
        if self.is_external:
            raise IndexError("External function does not have a body")
        return self.regions[0].blocks[0]

    def add_entry_block(self):
        """
        Add an entry block to the function body using the function signature to
        infer block arguments.
        Returns the newly created block
        """
        if not self.is_external:
            raise IndexError("The function already has an entry block!")
        self.body.blocks.append(*self.type.inputs)
        return self.body.blocks[0]

    @property
    def arg_attrs(self):
        return ArrayAttr(self.attributes[ARGUMENT_ATTRIBUTE_NAME])

    @arg_attrs.setter
    def arg_attrs(self, attribute: Union[ArrayAttr, list]):
        if isinstance(attribute, ArrayAttr):
            self.attributes[ARGUMENT_ATTRIBUTE_NAME] = attribute
        else:
            self.attributes[ARGUMENT_ATTRIBUTE_NAME] = ArrayAttr.get(
                attribute, context=self.context
            )

    @property
    def arguments(self):
        return self.entry_block.arguments

    @property
    def result_attrs(self):
        return self.attributes[RESULT_ATTRIBUTE_NAME]

    @result_attrs.setter
    def result_attrs(self, attribute: ArrayAttr):
        self.attributes[RESULT_ATTRIBUTE_NAME] = attribute
