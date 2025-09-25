#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from ._standalone_ops_gen import *
from .._mlir_libs._standaloneDialectsPybind11.standalone import *

from .._mlir_libs import get_dialect_registry as _get_dialect_registry
from .._mlir_libs._capi import register_dialect as _register_dialect

_dialect_registry = _get_dialect_registry()
if "quant" not in _dialect_registry.dialect_names:
    _register_dialect("quant", _dialect_registry)
