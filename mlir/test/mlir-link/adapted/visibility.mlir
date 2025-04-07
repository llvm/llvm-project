// RUN: mlir-link %s %p/Inputs/visibility.mlir -o - | FileCheck %s
// RUN: mlir-link %p/Inputs/visibility.mlir %s -o - | FileCheck %s

// mlir.alias not yet supported; comdat not yet supported
// XFAIL: *

module {
  llvm.comdat @__llvm_global_comdat {
    llvm.comdat_selector @c1 any
  }
// CHECK-DAG: llvm.mlir.global external hidden @v1(0 : i32)
  llvm.mlir.global external @v1(0 : i32) {addr_space = 0 : i32} : i32
// CHECK-DAG: llvm.mlir.global external protected @v2(0 : i32)
  llvm.mlir.global external @v2(0 : i32) {addr_space = 0 : i32} : i32
// CHECK-DAG: llvm.mlir.global external hidden @v3(0 : i32)
  llvm.mlir.global external protected @v3(0 : i32) {addr_space = 0 : i32, dso_local} : i32
// CHECK-DAG: llvm.mlir.global external hidden @v4(0 : i32)
  llvm.mlir.global external @v4(1 : i32) comdat(@__llvm_global_comdat::@c1) {addr_space = 0 : i32} : i32
// CHECK-DAG: llvm.mlir.alias external hidden @a1: i32 {
  llvm.mlir.alias external @a1 : i32 {
    %0 = llvm.mlir.addressof @v1 : !llvm.ptr
    llvm.return %0 : !llvm.ptr
  }
// CHECK-DAG: llvm.mlir.alias external protected @a2: i32 {
  llvm.mlir.alias external @a2 : i32 {
    %0 = llvm.mlir.addressof @v2 : !llvm.ptr
    llvm.return %0 : !llvm.ptr
  }
// CHECK-DAG: llvm.mlir.alias external hidden @a3: i32 {
  llvm.mlir.alias external protected @a3 {dso_local} : i32 {
    %0 = llvm.mlir.addressof @v3 : !llvm.ptr
    llvm.return %0 : !llvm.ptr
  }
// CHECK-DAG: llvm.func hidden @f1() {
  llvm.func @f1() {
    llvm.return
  }
// CHECK-DAG: llvm.func protected @f2() {
  llvm.func @f2() {
    llvm.return
  }
// CHECK-DAG: llvm.func hidden @f3() {
  llvm.func protected @f3() attributes {dso_local} {
    llvm.return
  }
}
