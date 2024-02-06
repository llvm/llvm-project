// RUN: mlir-opt %s -finalize-memref-to-llvm 2>&1 | FileCheck %s
// Since the error is at an unknown location, we use FileCheck instead of
// -veri-y-diagnostics here

// CHECK: conversion of memref memory space #gpu.address_space<workgroup> to integer address space failed. Consider adding memory space conversions
// CHECK-LABEL: @issue_70160
func.func @issue_70160() {
  %alloc = memref.alloc() : memref<1x32x33xi32, #gpu.address_space<workgroup>>
  %alloc1 = memref.alloc() : memref<i32>
  %c0 = arith.constant 0 : index
  // CHECK: memref.load
  %0 = memref.load %alloc[%c0, %c0, %c0] : memref<1x32x33xi32, #gpu.address_space<workgroup>>
  memref.store %0, %alloc1[] : memref<i32>
  func.return
}
