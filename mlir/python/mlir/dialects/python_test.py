#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from ._python_test_ops_gen import *
from .._mlir_libs._mlirPythonTest import (
    TestAttr,
    TestType,
    TestTensorValue,
    TestIntegerRankedTensorType,
)


def register_python_test_dialect(registry):
    from .._mlir_libs import _mlirPythonTest

    _mlirPythonTest.register_dialect(registry)
