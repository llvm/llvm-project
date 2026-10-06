// RUN: not tr-opt %s --split-input-file --tr-split-host-device 2>&1 | FileCheck %s

// Milestone 28: the split pass is a hard failure without a device module
// or without a host launch. Missing kernel symbols are already covered
// by gpu-module-symbols-invalid.mlir (SymbolUserOpInterface).

// CHECK: tr-split-host-device expects a gpu.module

func.func @no_device(%in: !tr.buffer<MxKxf32>, %out: !tr.buffer<Mxf32>) {
  return
}

// -----

// CHECK: tr-split-host-device expects at least one gpu.launch_func

module attributes {gpu.container_module} {
  gpu.module @tr_kernels {
    gpu.func @row_sum_kernel() kernel {
      gpu.return
    }
  }
}
