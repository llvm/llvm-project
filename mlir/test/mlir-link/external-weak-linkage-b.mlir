// RUN: mlir-link -sort-symbols -split-input-file %s | FileCheck %s

// CHECK: llvm.func extern_weak @bar() -> i32

// CHECK: llvm.func @foo() -> i32 {
// CHECK-NEXT:   %0 = llvm.call @bar() : () -> i32
// CHECK-NEXT:   llvm.return %0 : i32
// CHECK-NEXT: }

llvm.func extern_weak @bar() -> i32

llvm.func @foo() -> i32 {
  %0 = llvm.call @bar() : () -> i32
  llvm.return %0 : i32
}
