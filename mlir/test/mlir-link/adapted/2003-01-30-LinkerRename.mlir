// RUN: mlir-link %S/Inputs/2003-01-30-LinkerRename.mlir %s | FileCheck %s

// internal function not resolved properly
// XFAIL: *

// CHECK:       llvm.mlir.global external @bar() {addr_space = 0 : i32} : !llvm.ptr {
// CHECK-NEXT:    %0 = llvm.mlir.addressof @foo.2 : !llvm.ptr
// CHECK-NEXT:    llvm.return %0 : !llvm.ptr
// CHECK-NEXT:  }

// CHECK:      llvm.func internal @foo.2() -> i32 attributes {dso_local} {
// CHECK-NEXT:   %0 = llvm.mlir.constant(7 : i32) : i32
// CHECK-NEXT:   llvm.return %0 : i32
// CHECK-NEXT: }

// CHECK:      llvm.func @foo() -> i32 {
// CHECK-NEXT:   %0 = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT:   llvm.return %0 : i32
// CHECK-NEXT: }

module {
  llvm.func @foo() -> i32 {
    %0 = llvm.mlir.constant(0 : i32) : i32
    llvm.return %0 : i32
  }
}
