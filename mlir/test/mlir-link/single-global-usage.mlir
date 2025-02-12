// RUN: mlir-link -split-input-file %s | FileCheck %s

// CHECK:  llvm.mlir.global external @number(7 : i32) {addr_space = 0 : i32} : i32

// -----

llvm.mlir.global @number(7 : i32) : i32


// -----

llvm.mlir.global @number() : i32

