// RUN: mlir-opt %s -split-input-file -convert-gpu-to-rocdl | FileCheck %s --check-prefixes=CHECK,ROCDL
// RUN: mlir-opt %s -split-input-file -convert-gpu-to-nvvm | FileCheck %s --check-prefixes=CHECK,NVVM

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


// -----

gpu.module @kernel {
  gpu.func @workgroup(%arg0: f32) workgroup(%arg1: memref<4xf32, #gpu.address_space<workgroup>>) {
    %c0 = arith.constant 0 : index
    memref.store %arg0, %arg1[%c0] : memref<4xf32, #gpu.address_space<workgroup>>
    gpu.return
  }
}

// CHECK-LABEL:  llvm.func @workgroup
//       CHECK:  llvm.store
//  CHECK-SAME:   : !llvm.ptr<f32, 3>

// -----

gpu.module @kernel {
  gpu.func @nested_memref(%arg0: memref<4xmemref<4xf32, #gpu.address_space<global>>, #gpu.address_space<global>>) -> f32 {
    %c0 = arith.constant 0 : index
    %inner = memref.load %arg0[%c0] : memref<4xmemref<4xf32, #gpu.address_space<global>>, #gpu.address_space<global>>
    %value = memref.load %inner[%c0] : memref<4xf32, #gpu.address_space<global>>
    gpu.return %value : f32
  }
}

// CHECK-LABEL:  llvm.func @nested_memref
//       CHECK:  llvm.load
//  CHECK-SAME:   : !llvm.ptr<{{.*}}, 1>
//       CHECK: [[value:%.+]] = llvm.load
//  CHECK-SAME:   : !llvm.ptr<f32, 1>
//       CHECK: llvm.return [[value]]
