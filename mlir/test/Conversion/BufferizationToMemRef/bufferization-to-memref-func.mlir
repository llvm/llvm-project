// RUN: mlir-opt -verify-diagnostics --pass-pipeline="builtin.module(func.func(convert-bufferization-to-memref))" -split-input-file %s | FileCheck %s

// CHECK-NOT: func @deallocHelper
// CHECK-LABEL: func @conversion_dealloc_simple
// CHECK-SAME: [[ARG0:%.+]]: memref<2xf32>
// CHECK-SAME: [[ARG1:%.+]]: i1
func.func @conversion_dealloc_simple(%arg0: memref<2xf32>, %arg1: i1) {
  bufferization.dealloc (%arg0 : memref<2xf32>) if (%arg1)
  return
}

//      CHECk: scf.if [[ARG1]] {
// CHECk-NEXT:   memref.dealloc [[ARG0]] : memref<2xf32>
// CHECk-NEXT: }
// CHECk-NEXT: return

// -----

func.func @conversion_dealloc_multiple_memrefs_and_retained(%arg0: memref<2xf32>, %arg1: memref<5xf32>, %arg2: memref<1xf32>, %arg3: i1, %arg4: i1, %arg5: memref<2xf32>) -> (i1, i1) {
  // expected-error @below {{library function required for generic lowering, but cannot be automatically inserted when operating on functions}}
  // expected-error @below {{failed to legalize operation 'bufferization.dealloc' that was explicitly marked illegal}}
  %0:2 = bufferization.dealloc (%arg0, %arg1 : memref<2xf32>, memref<5xf32>) if (%arg3, %arg4) retain (%arg2, %arg5 : memref<1xf32>, memref<2xf32>)
  return %0#0, %0#1 : i1, i1
}
