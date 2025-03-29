// RUN: mlir-link -sort-symbols -split-input-file %s | FileCheck %s

// CHECK:       llvm.func @f1() -> i32 {
// CHECK-NEXT:    %0 = llvm.mlir.addressof @number : !llvm.ptr
// CHECK-NEXT:    %1 = llvm.load %0 : !llvm.ptr -> i32
// CHECK-NEXT:    llvm.return %1 : i32
// CHECK-NEXT:  }
// CHECK-NEXT:  llvm.mlir.global external @number(7 : i32)

llvm.mlir.global @number(7 : i32) : i32

// -----

llvm.mlir.global @number() {} : i32

llvm.func @f1() -> i32  {
  %0 = llvm.mlir.addressof @number : !llvm.ptr
  %1 = llvm.load %0 : !llvm.ptr -> i32
  llvm.return %1 : i32
}
