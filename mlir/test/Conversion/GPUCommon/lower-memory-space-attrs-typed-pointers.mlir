// RUN: mlir-opt %s -split-input-file -convert-gpu-to-rocdl='use-opaque-pointers=0' | FileCheck %s --check-prefixes=CHECK,ROCDL
// RUN: mlir-opt %s -split-input-file -convert-gpu-to-nvvm='use-opaque-pointers=0' | FileCheck %s --check-prefixes=CHECK,NVVM

gpu.module @kernel {
  gpu.func @private(%arg0: f32) private(%arg1: memref<4xf32, #gpu.address_space<private>>) {
    %c0 = arith.constant 0 : index
    memref.store %arg0, %arg1[%c0] : memref<4xf32, #gpu.address_space<private>>
    gpu.return
  }
}

// CHECK-LABEL:  llvm.func @private
//      CHECK:  llvm.store
// ROCDL-SAME:   : !llvm.ptr<f32, 5>
//  NVVM-SAME:   : !llvm.ptr<f32>
