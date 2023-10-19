#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from ._affine_ops_gen import *
from ._affine_ops_gen import _Dialect

try:
    from ..ir import *
    from ._ods_common import (
        get_op_result_or_value as _get_op_result_or_value,
        get_op_results_or_values as _get_op_results_or_values,
        _cext as _ods_cext,
    )
except ImportError as e:
    raise RuntimeError("Error loading imports from extension module") from e

from typing import Optional, Sequence, Union


@_ods_cext.register_operation(_Dialect, replace=True)
class AffineStoreOp(AffineStoreOp):
    """Specialization for the Affine store operation."""

    def __init__(
        self,
        value: Union[Operation, OpView, Value],
        memref: Union[Operation, OpView, Value],
        map: AffineMap = None,
        *,
        map_operands=None,
        loc=None,
        ip=None,
    ):
        """Creates an affine store operation.

        - `value`: the value to store into the memref.
        - `memref`: the buffer to store into.
        - `map`: the affine map that maps the map_operands to the index of the
          memref.
        - `map_operands`: the list of arguments to substitute the dimensions,
          then symbols in the affine map, in increasing order.
        """
        map = map if map is not None else []
        map_operands = map_operands if map_operands is not None else []
        indicies = [_get_op_result_or_value(op) for op in map_operands]
        _ods_successors = None
        super().__init__(
            value, memref, indicies, AffineMapAttr.get(map), loc=loc, ip=ip
        )
