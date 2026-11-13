// RUN: mlir-translate -mlir-to-llvmir -split-input-file %s | FileCheck %s

llvm.func @deref(%arg0: i64, %arg1: !llvm.ptr) {
  // CHECK: inttoptr {{.*}} !dereferenceable [[D0:![0-9]+]]
  %0 = llvm.inttoptr %arg0 dereferenceable<bytes = 4> : i64 to !llvm.ptr
  %1 = llvm.load %0 <alignment = 4> : !llvm.ptr -> i32
  // CHECK: load {{.*}} !dereferenceable [[D1:![0-9]+]]
  %2 = llvm.load %arg1 dereferenceable<bytes = 8> <alignment = 8> : !llvm.ptr -> !llvm.ptr
  llvm.store %1, %2 <alignment = 4> : i32, !llvm.ptr
  llvm.return
}

llvm.func @deref_or_null(%arg0: i64, %arg1: !llvm.ptr) {
  // CHECK: inttoptr {{.*}} !dereferenceable_or_null [[D0]]
  %0 = llvm.inttoptr %arg0 dereferenceable<bytes = 4, mayBeNull = true> : i64 to !llvm.ptr
  %1 = llvm.load %0 <alignment = 4> : !llvm.ptr -> i32
  // CHECK: load {{.*}} !dereferenceable_or_null [[D1]]
  %2 = llvm.load %arg1 dereferenceable<bytes = 8, mayBeNull = true> <alignment = 8> : !llvm.ptr -> !llvm.ptr
  llvm.store %1, %2 <alignment = 4> : i32, !llvm.ptr
  llvm.return
}

// Dereferenceable metadata is dropped if the cast is folded.
llvm.func @deref_folded_away() -> !llvm.ptr {
  // CHECK: define ptr @deref_folded_away()
  // CHECK-NEXT: ret ptr inttoptr (i64 64 to ptr)
  %0 = llvm.mlir.constant(64 : i64) : i64
  %1 = llvm.inttoptr %0 dereferenceable<bytes = 4> : i64 to !llvm.ptr
  llvm.return %1 : !llvm.ptr
}

// CHECK: [[D0]] = !{i64 4}
// CHECK: [[D1]] = !{i64 8}
