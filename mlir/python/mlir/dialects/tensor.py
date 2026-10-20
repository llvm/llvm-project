#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
from typing import Optional

from ._tensor_ops_gen import *
from ._tensor_ops_gen import _Dialect
from ..extras.meta import region_op

try:
    from ..ir import *
except ImportError as e:
    raise RuntimeError("Error loading imports from extension module") from e

from typing import Sequence, Union
from ._ods_common import _cext as _ods_cext
from ._ods_common import get_op_result_or_op_results as _get_op_result_or_op_results


@_ods_cext.register_operation(_Dialect, replace=True)
class EmptyOp(EmptyOp):
    """Extends the tensor.empty op."""

    def __init__(
        self,
        sizes: Sequence[Union[int, Value]],
        element_type: Type,
        *,
        encoding: Optional[Attribute] = None,
        loc=None,
        ip=None,
    ):
        """Constructs an `empty` with mixed static/dynamic sizes."""
        # TODO: Refactor the EmptyOp to take an element type attribute and
        # then use normal result type inference, unifying the Python and C++ side
        # with a standard mechanism (versus stashing that in builders).
        dynamic_sizes = []
        static_sizes = []
        for s in sizes:
            if isinstance(s, int):
                static_sizes.append(s)
            else:
                static_sizes.append(ShapedType.get_dynamic_size())
                dynamic_sizes.append(s)
        result_type = RankedTensorType.get(static_sizes, element_type, encoding)
        super().__init__(result_type, dynamic_sizes, loc=loc, ip=ip)


def empty(
    sizes: Sequence[Union[int, Value]],
    element_type: Type,
    *,
    encoding: Optional[Attribute] = None,
    loc=None,
    ip=None,
) -> _ods_cext.ir.Value:
    return _get_op_result_or_op_results(
        EmptyOp(
            sizes=sizes, element_type=element_type, encoding=encoding, loc=loc, ip=ip
        )
    )


generate = region_op(
    lambda result, dynamic_extents: GenerateOp(result, dynamic_extents),
    terminator=lambda args: YieldOp(args[0]),
)
