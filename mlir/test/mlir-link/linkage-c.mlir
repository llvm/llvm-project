// RUN: mlir-link -split-input-file %s | FileCheck %s

// CHECK: llvm.mlir.global common @test1_a(0 : i8) {addr_space = 0 : i32} : i8
// CHECK: llvm.mlir.global external @test2_a(0 : i8) {addr_space = 0 : i32} : i8
// CHECK: llvm.mlir.global common @test3_a(0 : i16) {addr_space = 0 : i32} : i16
// CHECK: llvm.mlir.global common @test4_a(0 : i16) {addr_space = 0 : i32, alignment = 8 : i64} : i16

llvm.mlir.global weak @test1_a(1 : i8) {addr_space = 0 : i32} : i8
llvm.mlir.global external @test2_a() {addr_space = 0 : i32} : i8
llvm.mlir.global common @test3_a(0 : i16) {addr_space = 0 : i32} : i16
llvm.mlir.global common @test4_a(0 : i16) {addr_space = 0 : i32, alignment = 4 : i64} : i16

// -----

llvm.mlir.global common @test1_a(0 : i8) {addr_space = 0 : i32} : i8
llvm.mlir.global external @test2_a(0 : i8) {addr_space = 0 : i32} : i8
llvm.mlir.global common @test3_a(0 : i8) {addr_space = 0 : i32} : i8
llvm.mlir.global common @test4_a(0 : i8) {addr_space = 0 : i32, alignment = 8 : i64} : i8
