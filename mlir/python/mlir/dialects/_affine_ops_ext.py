#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

try:
    from ..ir import *
    from ._ods_common import get_op_result_or_value as _get_op_result_or_value
    from ._ods_common import get_op_results_or_values as _get_op_results_or_values
except ImportError as e:
    raise RuntimeError("Error loading imports from extension module") from e

from typing import Optional, Sequence, Union


class AffineStoreOp:
    """Specialization for the Affine store operation."""

    def __init__(
        self,
        value: Value,
        memref: Union[Operation, OpView, Value],
        map,
        *,
        map_operands=[],
        loc=None,
        ip=None
    ):
        """Creates an affine load operation.

        - `value`: the value to store into the memref.
        - `memref`: the buffer to store into.
        - `map`: the affine map that maps the map_operands to the index of the 
          memref.
        - `map_operands`: the list of arguments to substitute the dimensions, 
          then symbols in the affine map, in increasing order.
        """
        operands = [
            _get_op_result_or_value(value),
            _get_op_result_or_value(memref),
            *[_get_op_result_or_value(op) for op in map_operands]
        ]
        results = []
        attributes = {"map": AffineMapAttr.get(map)}
        regions = None
        _ods_successors = None
        super().__init__(self.build_generic(
            attributes=attributes,
            results=results,
            operands=operands,
            successors=_ods_successors,
            regions=regions,
            loc=loc,
            ip=ip
        ))
