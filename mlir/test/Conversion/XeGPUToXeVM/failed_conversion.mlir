// RUN: mlir-opt --convert-xegpu-to-xevm %s -split-input-file -verify-diagnostics

// Verify that xegpu.dpas with unsupported element types (i16) is rejected
// during XeGPUToXeVM conversion rather than crashing.

gpu.module @test_kernel [#xevm.target<chip = "pvc">] {
  func.func @main() {
    %0 = arith.constant dense<0> : vector<4xi16>
    %1 = arith.constant dense<0> : vector<4xi32>
    // expected-error@+1 {{failed to legalize operation 'xegpu.dpas' that was explicitly marked illegal}}
    %2 = xegpu.dpas %0, %0, %1 : vector<4xi16>, vector<4xi16>, vector<4xi32> -> vector<4xi32>
    return
  }
}
