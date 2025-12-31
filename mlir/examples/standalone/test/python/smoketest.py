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


# CHECK: Testing mlir package
print("Testing mlir package", file=sys.stderr)

import mlir.ir
from mlir.dialects import (
    amdgpu,
    gpu,
    irdl,
    llvm,
    nvgpu,
    pdl,
    quant,
    smt,
    sparse_tensor,
    transform,
    # Note: uncommenting linalg below will cause
    # LLVM ERROR: Attempting to attach an interface to an unregistered operation builtin.unrealized_conversion_cast.
    # unless you have built both mlir and mlir_standalone with
    # -DCMAKE_C_VISIBILITY_PRESET=hidden -DCMAKE_CXX_VISIBILITY_PRESET=hidden -DCMAKE_VISIBILITY_INLINES_HIDDEN=ON
    # which is highly recommended.
    # linalg,
)

# CHECK-NOT: RuntimeWarning: nanobind: type '{{.*}}' was already registered!

with mlir.ir.Context():
    mlir_module = mlir.ir.Module.parse(
        """
    %0 = arith.constant 3 : i32
    """
    )
    # CHECK: %[[C3:.*]] = arith.constant 3 : i32
    print(str(mlir_module), file=sys.stderr)
