// RUN: mlir-opt %s --gpu-to-llvm="use-bare-pointers-for-kernels=1" -split-input-file -verify-diagnostics

// Test that gpu.launch_func with an unranked memref kernel argument fails
// gracefully (no crash) when using the bare-pointer calling convention.
// See issue #184939.

module attributes {gpu.container_module} {
  gpu.module @kernels {
    gpu.func @kernel(%arg0: memref<*xi32>) kernel {
      gpu.return
    }
  }
  func.func @main(%arg: memref<*xi32>) {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : i32
    // expected-error@+1 {{failed to legalize operation 'gpu.launch_func'}}
    gpu.launch_func @kernels::@kernel
        blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1)
        dynamic_shared_memory_size %c0
        args(%arg : memref<*xi32>)
    return
  }
}
