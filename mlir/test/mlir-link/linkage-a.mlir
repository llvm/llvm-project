// RUN: mlir-link -split-input-file %s | FileCheck %s

// CHECK: llvm.mlir.global external @X() {addr_space = 0 : i32} : i32

// CHECK: llvm.func @bar()

// CHECK: llvm.func @foo() -> i32

llvm.mlir.global linkonce @X(5 : i32) {addr_space = 0 : i32} : i32
llvm.func linkonce @foo() -> i32 {
  %0 = llvm.mlir.constant(7 : i32) : i32
  llvm.return %0 : i32
}

// -----

llvm.mlir.global external @X() {addr_space = 0 : i32} : i32
llvm.func @foo() -> i32
llvm.func @bar() {
  %0 = llvm.mlir.addressof @X : !llvm.ptr
  %1 = llvm.load %0 {alignment = 4 : i64} : !llvm.ptr -> i32
  %2 = llvm.call @foo() : () -> i32
  llvm.return
}
