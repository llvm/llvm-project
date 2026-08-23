// RUN: not tr-opt %s 2>&1 | FileCheck %s

// Milestone 20: nested SymbolRefAttr lookup is a hard failure when the
// kernel symbol is missing. gpu.launch_func uses SymbolUserOpInterface.

module attributes {gpu.container_module} {
  gpu.module @tr_kernels {
    gpu.func @other_kernel() kernel {
      gpu.return
    }
  }
  func.func @host(%in: memref<?x?xf32>, %out: memref<?xf32>) {
    %c1 = arith.constant 1 : index
    %c256 = arith.constant 256 : index
    // CHECK: kernel function '@tr_kernels::@row_sum_kernel' is undefined
    gpu.launch_func @tr_kernels::@row_sum_kernel
        blocks in (%c1, %c1, %c1) threads in (%c256, %c1, %c1)
        args(%in : memref<?x?xf32>, %out : memref<?xf32>)
    return
  }
}
