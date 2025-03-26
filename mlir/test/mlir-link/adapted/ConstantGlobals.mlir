// RUN: mlir-link %s %S/Inputs/ConstantGlobals.mlir -o -| FileCheck %s
// RUN: mlir-link %S/Inputs/ConstantGlobals.mlir %s -o - | FileCheck %s

module {
// CHECK-DAG: llvm.mlir.global external constant @X(dense<8> : tensor<1xi32>)
  llvm.mlir.global external @X() {addr_space = 0 : i32} : !llvm.array<1 x i32>
// CHECK-DAG: llvm.mlir.global external @Y() {addr_space = 0 : i32} : !llvm.array<1 x i32>
  llvm.mlir.global external @Y() {addr_space = 0 : i32} : !llvm.array<1 x i32>
  llvm.func @"use-Y"() -> !llvm.ptr {
    %0 = llvm.mlir.addressof @Y : !llvm.ptr
    llvm.return %0 : !llvm.ptr
  }
}
