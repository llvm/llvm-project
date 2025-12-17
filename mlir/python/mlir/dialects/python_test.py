#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from ._python_test_ops_gen import *


def register_python_test_dialect(registry, use_nanobind):
    if use_nanobind:
        from .._mlir_libs import _mlirPythonTestNanobind

        _mlirPythonTestNanobind.register_dialect(registry)
    else:
        from .._mlir_libs import _mlirPythonTestPybind11

        _mlirPythonTestPybind11.register_dialect(registry)
