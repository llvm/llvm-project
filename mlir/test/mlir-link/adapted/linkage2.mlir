// RUN: mlir-link %s %p/Inputs/linkage2.mlir -o - | FileCheck %s
// RUN: mlir-link %p/Inputs/linkage2.ll %s -o - | FileCheck %s

// broken alignment resolution
// XFAIL: *

module {
  llvm.mlir.global common @test1_a(0 : i8) {addr_space = 0 : i32} : i8
// CHECK-DAG: llvm.mlir.global common @test1_a(0 : i8)
  llvm.mlir.global external @test2_a(0 : i8) {addr_space = 0 : i32} : i8
// CHECK-DAG: llvm.mlir.global external @test2_a(0 : i8)
  llvm.mlir.global common @test3_a(0 : i8) {addr_space = 0 : i32} : i8
// CHECK-DAG: llvm.mlir.global common @test3_a(0 : i16)
  llvm.mlir.global common @test4_a(0 : i8) {addr_space = 0 : i32, alignment = 8 : i64} : i8
// CHECK-DAG: llvm.mlir.global common @test4_a(0 : i16) {addr_space = {[0-9]+} : i32, alignment = 8 : i64}
}
