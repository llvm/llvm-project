# RUN: %python %s nanobind | FileCheck %s

from mlir_standalone.ir import *
from mlir_standalone.dialects import standalone_nanobind as standalone_d

with Context():
    standalone_d.register_dialects()
    module = Module.parse(
        """
    %0 = arith.constant 2 : i32
    %1 = standalone.foo %0 : i32
    """
    )
    # CHECK: %[[C:.*]] = arith.constant 2 : i32
    # CHECK: standalone.foo %[[C]] : i32
    print(str(module))
