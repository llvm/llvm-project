# RUN: %python %s 2>&1 | FileCheck %s
import sys

# CHECK: Testing mlir_standalone package
print("Testing mlir_standalone package", file=sys.stderr)

import mlir_standalone.ir
from mlir_standalone.dialects import standalone_nanobind as standalone_d

with mlir_standalone.ir.Context():
    standalone_d.register_dialects()
    standalone_module = mlir_standalone.ir.Module.parse(
        """
    %0 = arith.constant 2 : i32
    %1 = standalone.foo %0 : i32
    """
    )
    # CHECK: %[[C2:.*]] = arith.constant 2 : i32
    # CHECK: standalone.foo %[[C2]] : i32
    print(str(standalone_module), file=sys.stderr)

    custom_type = standalone_d.CustomType.get("foo")
    # CHECK: !standalone.custom<"foo">
    print(custom_type, file=sys.stderr)

    # CHECK: this is a fp16 type
    standalone_d.print_fp_type(mlir_standalone.ir.F16Type.get(), sys.stderr)
    # CHECK: this is a fp32 type
    standalone_d.print_fp_type(mlir_standalone.ir.F32Type.get(), sys.stderr)
    # CHECK: this is a fp64 type
    standalone_d.print_fp_type(mlir_standalone.ir.F64Type.get(), sys.stderr)


# CHECK: Testing mlir package
print("Testing mlir package", file=sys.stderr)

from mlir.ir import *

# CHECK-NOT: RuntimeWarning: nanobind: type '{{.*}}' was already registered!
