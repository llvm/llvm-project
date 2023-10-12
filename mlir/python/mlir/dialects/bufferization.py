#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from ._bufferization_ops_gen import *
from ._bufferization_ops_gen import _Dialect
from ._bufferization_enum_gen import *

try:
    from typing import Sequence, Union
    from ..ir import *
    from ._ods_common import get_default_loc_context, _cext as _ods_cext

    from typing import Any, List, Union
except ImportError as e:
    raise RuntimeError("Error loading imports from extension module") from e

_AllocTensorOp = AllocTensorOp


@_ods_cext.register_operation(_Dialect, replace=True)
class AllocTensorOp(_AllocTensorOp):
    """Extends the bufferization.alloc_tensor op."""

    def __init__(
        self,
        tensor_type: Type,
        dynamic_sizes: Sequence[Value],
        copy: Value,
        size_hint: Value,
        escape: BoolAttr,
        *,
        loc=None,
        ip=None,
    ):
        """Constructs an `alloc_tensor` with static and/or dynamic sizes."""
        context = get_default_loc_context(loc)
        attributes = {}
        if escape:
            attributes["escape"] = escape
        super(_AllocTensorOp, self).__init__(
            self.build_generic(
                results=[tensor_type],
                operands=[dynamic_sizes, copy, size_hint],
                attributes=attributes,
                loc=loc,
                ip=ip,
            )
        )
