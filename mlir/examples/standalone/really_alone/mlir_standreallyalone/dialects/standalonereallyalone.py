#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from ._ods_common import _cext

_cext.globals.append_dialect_search_prefix("mlir_standreallyalone.dialects")

from ._standalone_ops_gen import *
from .._mlir_libs._standReallyAloneDialectsNanobind.standalone import *
