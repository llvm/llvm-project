// RUN: mlir-opt --canonicalize -split-input-file %s | FileCheck %s

// CHECK-LABEL: func @fold_memref2pointer_pointer2memref(
// CHECK-SAME:    %[[PTR:.*]]: !llvm.ptr
func.func @fold_memref2pointer_pointer2memref(%ptr: !llvm.ptr) -> !llvm.ptr {
  // CHECK-NOT: polygeist.pointer2memref
  // CHECK-NOT: polygeist.memref2pointer
  // CHECK: return %[[PTR]]
  %memref = "polygeist.pointer2memref"(%ptr) : (!llvm.ptr) -> memref<?xf64>
  %ptr2 = "polygeist.memref2pointer"(%memref) : (memref<?xf64>) -> !llvm.ptr
  func.return %ptr2 : !llvm.ptr
}

// -----

// CHECK-LABEL: func @fold_memref2pointer_cast(
// CHECK-SAME:    %[[MEMREF:.*]]: memref<10xf64>
func.func @fold_memref2pointer_cast(%memref: memref<10xf64>) -> !llvm.ptr {
  // CHECK: %[[RES:.*]] = "polygeist.memref2pointer"(%[[MEMREF]])
  // CHECK: return %[[RES]]
  %cast = memref.cast %memref : memref<10xf64> to memref<?xf64>
  %ptr = "polygeist.memref2pointer"(%cast) : (memref<?xf64>) -> !llvm.ptr
  func.return %ptr : !llvm.ptr
}
