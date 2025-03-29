// RUN: mlir-link -sort-symbols -split-input-file %s | FileCheck %s

// CHECK:       llvm.func @f1()
// CHECK:       llvm.func @f2() {
// CHECK-NEXT:    llvm.call @f1() : () -> ()
// CHECK-NEXT:    llvm.return
// CHECK-NEXT:  }

llvm.func @f1()

llvm.func @f2() {
  llvm.call @f1() : () -> ()
  llvm.return
}

// -----

llvm.func @f1()
