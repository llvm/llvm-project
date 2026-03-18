// RUN: mlir-opt -pass-pipeline="builtin.module(gpu.module(convert-gpu-to-llvm-spv))" -verify-diagnostics %s

module attributes {gpu.container_module} {
  gpu.module @kernels {
    // expected-error @below {{failed to legalize operation 'gpu.barrier' that was explicitly marked illegal}}
    gpu.barrier
  }
}
