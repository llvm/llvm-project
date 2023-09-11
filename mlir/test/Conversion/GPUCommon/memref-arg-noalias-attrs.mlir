// RUN: mlir-opt %s -split-input-file -convert-gpu-to-rocdl='use-opaque-pointers=1 use-bare-ptr-memref-call-conv=1' | FileCheck %s --check-prefixes=CHECK,ROCDL
// RUN: mlir-opt %s -split-input-file -convert-gpu-to-nvvm='use-opaque-pointers=1 use-bare-ptr-memref-call-conv=1' | FileCheck %s --check-prefixes=CHECK,NVVM

gpu.module @kernel {
  gpu.func @func_with_noalias_attr(%arg0 : memref<f32> {llvm.noalias} ) {
    gpu.return
  }
}

// CHECK-LABEL:  llvm.func @func_with_noalias_attr
// ROCDL-SAME:  !llvm.ptr {llvm.noalias}
//  NVVM-SAME:  !llvm.ptr {llvm.noalias}


// -----

gpu.module @kernel {
  gpu.func @func_without_any_attr(%arg0 : memref<f32> ) {
    gpu.return
  }
}

// CHECK-LABEL:  llvm.func @func_without_any_attr
// ROCDL-SAME:  !llvm.ptr
//  NVVM-SAME:  !llvm.ptr
