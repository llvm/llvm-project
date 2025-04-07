// RUN: mlir-link %s -split-input-file -o - | FileCheck %s

// The values in this chunk are strong the ones in the second chunk are weak,
// but we should still get the visibility from them.

// CHECK-DAG: llvm.mlir.global external hidden @v1(0 : i32)
llvm.mlir.global external @v1(0 : i32) {addr_space = 0 : i32} : i32
// CHECK-DAG: llvm.mlir.global external protected @v2(0 : i32)
llvm.mlir.global external @v2(0 : i32) {addr_space = 0 : i32} : i32
// CHECK-DAG: llvm.mlir.global external hidden @v3(0 : i32)
llvm.mlir.global external protected @v3(0 : i32) {addr_space = 0 : i32, dso_local} : i32

// CHECK-DAG: llvm.func hidden @f1()
llvm.func @f1() {
  llvm.return
}

// CHECK-DAG: llvm.func protected @f2()
llvm.func @f2() {
  llvm.return
}

// CHECK-DAG: llvm.func hidden @f3()
llvm.func protected @f3() {
  llvm.return
}

// -----

llvm.mlir.global weak hidden @v1(0 : i32) {addr_space = 0 : i32, dso_local} : i32
llvm.mlir.global weak protected @v2(0 : i32) {addr_space = 0 : i32, dso_local} : i32
llvm.mlir.global weak hidden @v3(0 : i32) {addr_space = 0 : i32, dso_local} : i32

llvm.func weak hidden @f1() {
  llvm.return
}

llvm.func weak protected @f2() {
  llvm.return
}

llvm.func weak hidden @f3() {
  llvm.return
}
