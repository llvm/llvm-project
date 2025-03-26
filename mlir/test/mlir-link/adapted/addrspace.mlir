// RUN: mlir-link %s -o - | FileCheck %s
// REQUIRES: func-addr-space-supp

module {
  llvm.mlir.global external @G(256 : i32) {addr_space = 2 : i32} : i32
  // CHECK: @G {{.*}} addr_space = 2
  llvm.mlir.alias @GA {addr_space = 2 : i32 } : !llvm.ptr {
  // CHECK: @GA {{.*}} addr_space = 2
      %0 = llvm.mlir.addressof @G : !llvm.ptr
      llvm.return %0 : !llvm.ptr
  }
  llvm.func @foo() {addr_space = 3 : i32} {
  // CHECK: @foo {{.*}} addr_space = 3
    llvm.return
  }
}
