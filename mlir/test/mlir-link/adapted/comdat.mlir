// RUN: mlir-link %s %p/Inputs/comdat.mlir -o - | FileCheck %s

// unimplemented conflict resolution
// XFAIL: *

module {
  llvm.comdat @__llvm_global_comdat {
    llvm.comdat_selector @foo largest
    llvm.comdat_selector @qux largest
    llvm.comdat_selector @any any
  }
  llvm.mlir.global external @foo(42 : i32) comdat(@__llvm_global_comdat::@foo) {addr_space = 0 : i32} : i32
  llvm.mlir.global external @qux(12 : i64) comdat(@__llvm_global_comdat::@qux) {addr_space = 0 : i32} : i64
  llvm.mlir.global external @any(6 : i64) comdat(@__llvm_global_comdat::@any) {addr_space = 0 : i32} : i64
  llvm.func @bar() -> i32 comdat(@__llvm_global_comdat::@foo) {
    %0 = llvm.mlir.constant(42 : i32) : i32
    llvm.return %0 : i32
  }
  llvm.func @baz() -> i32 comdat(@__llvm_global_comdat::@qux) {
    %0 = llvm.mlir.constant(12 : i32) : i32
    llvm.return %0 : i32
  }
}

// CHECK-DAG: llvm.comdat_selector @qux largest
// CHECK-DAG: llvm.comdat_selector @foo comdat largest
// CHECK-DAG: llvm.comdat_selector @any any

// CHECK-DAG: llvm.mlir.global @foo(43 : i64) comdat{{$}}
// CHECK-DAG: llvm.mlir.global @qux(12 : i64) comdat{{$}}
// CHECK-DAG: llvm.mlir.global @any(6 : i64) comdat{{$}}
// CHECK-NOT: llvm.mlir.global @in_unselected_group(13 : i32) comdat(@__llvm_global_comdat::@qux)

// CHECK-DAG: llvm.func @baz() -> i32 comdat(@__llvm_global_comdat::@qux) {
// CHECK-DAG: llvm.func @bar() -> i32 comdat(@__llvm_global_comdat::@foo) {
