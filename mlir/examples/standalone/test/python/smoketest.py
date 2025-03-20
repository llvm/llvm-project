# RUN: %python %s pybind11 | FileCheck %s
# RUN: %python %s nanobind | FileCheck %s

import sys
from mlir_standalone.ir import *
from mlir_standalone.dialects import builtin as builtin_d

if sys.argv[1] == "pybind11":
    from mlir_standalone.dialects import standalone_pybind11 as standalone_d
elif sys.argv[1] == "nanobind":
    from mlir_standalone.dialects import standalone_nanobind as standalone_d
else:
    raise ValueError("Expected either pybind11 or nanobind as arguments")


with Context():
    standalone_d.register_dialect()
    module = Module.parse(
        """
    %0 = arith.constant 2 : i32
    %1 = standalone.foo %0 : i32
    """
    )
    # CHECK: %[[C:.*]] = arith.constant 2 : i32
    # CHECK: standalone.foo %[[C]] : i32
    print(str(module))
