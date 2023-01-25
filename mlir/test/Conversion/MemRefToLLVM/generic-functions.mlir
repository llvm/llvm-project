// RUN: mlir-opt -pass-pipeline="builtin.module(convert-memref-to-llvm{use-generic-functions=1})" -split-input-file %s \
// RUN: | FileCheck %s --check-prefix="CHECK-NOTALIGNED"

// RUN: mlir-opt -pass-pipeline="builtin.module(convert-memref-to-llvm{use-generic-functions=1 use-aligned-alloc=1})" -split-input-file %s \
// RUN: | FileCheck %s --check-prefix="CHECK-ALIGNED"

// CHECK-LABEL: func @alloc()
func.func @zero_d_alloc() -> memref<f32> {
// CHECK-NOTALIGNED: llvm.call @_mlir_memref_to_llvm_alloc(%{{.*}}) : (i64) -> !llvm.ptr<i8>
// CHECK-ALIGNED: llvm.call @_mlir_memref_to_llvm_aligned_alloc(%{{.*}}, %{{.*}}) : (i64, i64) -> !llvm.ptr<i8>
  %0 = memref.alloc() : memref<f32>
  return %0 : memref<f32>
}

// -----

// CHECK-LABEL: func @dealloc()
func.func @dealloc(%arg0: memref<f32>) {
// CHECK-NOTALIGNED: llvm.call @_mlir_memref_to_llvm_free(%{{.*}}) : (!llvm.ptr<i8>) -> ()
// CHECK-ALIGNED: llvm.call @_mlir_memref_to_llvm_free(%{{.*}}) : (!llvm.ptr<i8>) -> ()
  memref.dealloc %arg0 : memref<f32>
  return
}
