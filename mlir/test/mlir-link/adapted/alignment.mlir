// RUN: mlir-link --sort-symbols %p/alignment.mlir %p/Inputs/alignment.mlir | FileCheck %s
// RUN: mlir-link --sort-symbols %p/Inputs/alignment.mlir %p/alignment.mlir | FileCheck %s

// alignment not yet resolved
// XFAIL: *

module {
  llvm.mlir.global weak @A(7 : i32) {addr_space = 0 : i32, alignment = 4 : i64} : i32
  // CHECK: llvm.mlir.global @A(7 : i32) {addr_space = 0 : i32, alignment = 8 : i32}
  llvm.mlir.global weak @B(7 : i32) {addr_space = 0 : i32, alignment = 8 : i64} : i32
  // CHECK: llvm.mlir.global @B(7 : i32) {addr_space = 0 : i32, alignment = 4 : i32}
  llvm.func weak @C() attributes {alignment = 4 : i64} {
    llvm.return
  }
  // CHECK: llvm.func @C() attributes {alignment = 8 : i64} {
  llvm.func weak @D() attributes {alignment = 8 : i64} {
    llvm.return
  }
  // CHECK: llvm.func @D() attributes {alignment = 4 : i64} {
  llvm.mlir.global common @E(0 : i32) {addr_space = 0 : i32, alignment = 4 : i64} : i32
  // CHECK: llvm.mlir.global common @E(7 : i32) {addr_space = 0 : i32, alignment = 8 : i32}
}
