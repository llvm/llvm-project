// RUN: mlir-opt -split-input-file -convert-xegpu-to-xevm -verify-diagnostics %s

// A result with more than one non-unit dimension cannot be flattened into the
// single contiguous access this lowering emits. Report it instead of tripping
// an assertion.
// See https://github.com/llvm/llvm-project/issues/208902.

gpu.module @test_kernel [#xevm.target<chip = "pvc">] {
  gpu.func @load_matrix_2d_result(%arg0: memref<4x8xf32>) kernel {
    %alloca = memref.alloca() : memref<4x8xf32, 3>
    %mdesc = xegpu.create_mem_desc %alloca : memref<4x8xf32, 3> -> !xegpu.mem_desc<4x8xf32>
    // expected-error@+1 {{failed to legalize operation 'xegpu.load_matrix'}}
    %res = xegpu.load_matrix %mdesc[0, 0] : !xegpu.mem_desc<4x8xf32> -> vector<4x8xf32>
    %c0 = arith.constant 0 : index
    vector.store %res, %arg0[%c0, %c0] : memref<4x8xf32>, vector<4x8xf32>
    gpu.return
  }
}
