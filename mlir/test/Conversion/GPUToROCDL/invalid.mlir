// RUN: mlir-opt %s -convert-gpu-to-rocdl -split-input-file --verify-diagnostics


// Demonstrate the need to register the cf and memref dialect as dependent.
// CHECK-LABEL: @dependentDialect
gpu.module @module {
  gpu.func @dependentDialect() {
    %arg0 = arith.constant 1 : i32
    // expected-error@+1 {{failed to legalize operation 'gpu.shuffle' that was explicitly marked illega}}
    %result = gpu.all_reduce %arg0 uniform {
    ^bb(%lhs : i32, %rhs : i32):
      %xor = arith.xori %lhs, %rhs : i32
      "gpu.yield"(%xor) : (i32) -> ()
    } : (i32) -> (i32)
    gpu.return
  }
}
