// RUN: mlir-link -sort-symbols -split-input-file %s | FileCheck %s

// CHECK: llvm.mlir.global common @a(0 : i8) {addr_space = 0 : i32} : i8
// CHECK: llvm.mlir.global common @b(0 : i8) {addr_space = 0 : i32} : i8
// CHECK: llvm.mlir.global common @c(0 : i8) {addr_space = 0 : i32} : i8
// CHECK: llvm.mlir.global common @d(0 : i8) {addr_space = 0 : i32} : i8
// CHECK: llvm.mlir.global common @e(0 : i8) {addr_space = 0 : i32} : i8
// CHECK: llvm.mlir.global common @f(0 : i8) {addr_space = 0 : i32} : i8
// CHECK: llvm.mlir.global external @g(1 : i8) {addr_space = 0 : i32} : i8

llvm.mlir.global linkonce @a(1 : i8) {addr_space = 0 : i32} : i8
llvm.mlir.global linkonce_odr @b(1 : i8) {addr_space = 0 : i32} : i8
llvm.mlir.global weak @c(1 : i8) {addr_space = 0 : i32} : i8
llvm.mlir.global weak_odr @d(1 : i8) {addr_space = 0 : i32} : i8
llvm.mlir.global internal @e(1 : i8) {addr_space = 0 : i32} : i8
llvm.mlir.global private @f(1 : i8) {addr_space = 0 : i32} : i8
llvm.mlir.global external @g(1 : i8) {addr_space = 0 : i32} : i8

// -----

llvm.mlir.global common @a(0 : i8) {addr_space = 0 : i32} : i8
llvm.mlir.global common @b(0 : i8) {addr_space = 0 : i32} : i8
llvm.mlir.global common @c(0 : i8) {addr_space = 0 : i32} : i8
llvm.mlir.global common @d(0 : i8) {addr_space = 0 : i32} : i8
llvm.mlir.global common @e(0 : i8) {addr_space = 0 : i32} : i8
llvm.mlir.global common @f(0 : i8) {addr_space = 0 : i32} : i8
llvm.mlir.global common @g(0 : i8) {addr_space = 0 : i32} : i8
