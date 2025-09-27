# RUN: %python %s pybind11 | FileCheck %s
# RUN: %python %s nanobind | FileCheck %s

import sys
from mlir_standalone.ir import *

if sys.argv[1] == "pybind11":
    from mlir_standalone.dialects import standalone_pybind11 as standalone_d
elif sys.argv[1] == "nanobind":
    from mlir_standalone.dialects import standalone_nanobind as standalone_d
else:
    raise ValueError("Expected either pybind11 or nanobind as arguments")


with Context():
    print("hello1", file=sys.stderr)
    standalone_d.register_dialects()
    print("hello2", file=sys.stderr)
    module = Module.parse(
        """
    %0 = arith.constant 2 : i32
    %1 = standalone.foo %0 : i32
    """
    )
    print("hello3", file=sys.stderr)
    # CHECK: %[[C:.*]] = arith.constant 2 : i32
    # CHECK: standalone.foo %[[C]] : i32
    print(str(module))
    print("hello4", file=sys.stderr)
