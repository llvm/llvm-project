// RUN: mlir-opt %s -finalize-memref-to-llvm 2>&1 | FileCheck %s
// Since the error is at an unknown location, we use FileCheck instead of
// -verify-diagnostics here

// CHECK: conversion of memref memory space #spirv.storage_class<StorageBuffer> to integer address space failed. Consider adding memory space conversions.
// CHECK-LABEL: @issue_131439
func.func @issue_131439(%arg0 : memref<i32>) {
  // CHECK: memref.alloca
  %alloca = memref.alloca() : memref<1x32x33xi32, #spirv.storage_class<StorageBuffer>>
  %c0 = arith.constant 0 : index
  // CHECK: memref.load
  %0 = memref.load %alloca[%c0, %c0, %c0] : memref<1x32x33xi32, #spirv.storage_class<StorageBuffer>>
  memref.store %0, %arg0[] : memref<i32>
  func.return
}
