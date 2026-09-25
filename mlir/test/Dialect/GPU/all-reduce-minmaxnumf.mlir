// RUN: mlir-opt --convert-gpu-to-nvvm %s | FileCheck %s

gpu.module @kernels {
  // CHECK-LABEL: llvm.func @minimum_number
  // CHECK: llvm.intr.minimumnum
  // CHECK: llvm.return
  gpu.func @minimum_number(%arg0 : f32, %out : memref<f32>) kernel {
    %result = gpu.all_reduce minimumnumf %arg0 uniform {} : (f32) -> f32
    memref.store %result, %out[] : memref<f32>
    gpu.return
  }

  // CHECK-LABEL: llvm.func @maximum_number
  // CHECK: llvm.intr.maximumnum
  // CHECK: llvm.return
  gpu.func @maximum_number(%arg0 : f32, %out : memref<f32>) kernel {
    %result = gpu.all_reduce maximumnumf %arg0 uniform {} : (f32) -> f32
    memref.store %result, %out[] : memref<f32>
    gpu.return
  }
}
