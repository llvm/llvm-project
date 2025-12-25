# RUN: %python %s 2>&1 | FileCheck %s

from mlir_standalone.ir import *
from mlir_standalone.dialects import standalone_nanobind as standalone_d

# CHECK-NOT: RuntimeWarning: nanobind: type '{{.*}}' was already registered!

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

    custom_type = standalone_d.CustomType.get("foo")
    # CHECK: !standalone.custom<"foo">
    print(custom_type)

from mlir.ir import *
