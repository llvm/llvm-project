// REQUIRES: xevm-conversions
// RUN: mlir-opt %s --gpu-lower-to-xevm-pipeline | FileCheck %s

module attributes {gpu.container_module} {
  // CHECK-LABEL: module attributes {gpu.container_module} {
  // CHECK: gpu.binary @kernels
  // CHECK-NOT: gpu.module @kernels
  gpu.module @kernels {
    func.func private @__hipblaslt_init_kernel(
        %arg0: memref<?xf32, 1>, %arg1: index, %arg2: index, %arg3: index,
        %arg4: index, %arg5: index) attributes {gpu.kernel}
    func.func private @__hipblaslt_init_small_kernel(
        %arg0: memref<?xf32, 1>, %arg1: index, %arg2: index, %arg3: index,
        %arg4: index, %arg5: index) attributes {gpu.kernel}
    func.func private @__hipblaslt_init_nan_tri_kernel(
        %arg0: memref<?xf32, 1>, %arg1: index, %arg2: index, %arg3: index,
        %arg4: index, %arg5: index, %arg6: i1) attributes {gpu.kernel}
  }
}
