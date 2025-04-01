// RUN: mlir-link -sort-symbols -split-input-file %s | FileCheck %s

// CHECK: llvm.func @bar() -> i32 {
// CHECK-NEXT:   %0 = llvm.call @foo() : () -> i32
// CHECK-NEXT:   llvm.return %0 : i32
// CHECK-NEXT: }

// CHECK: llvm.func @foo() -> i32 {
// CHECK-NEXT:   %0 = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT:   llvm.return %0 : i32
// CHECK-NEXT: }

llvm.func @foo() -> i32 {
  %0 = llvm.mlir.constant(1 : i32) : i32
  llvm.return %0 : i32
}

// -----

llvm.func weak @foo() -> i32 {
  %0 = llvm.mlir.constant(0 : i32) : i32
  llvm.return %0 : i32
}

llvm.func @bar() -> i32 {
  %0 = llvm.call @foo() : () -> i32
  llvm.return %0 : i32
}
