// RUN: mlir-opt %s -finalize-memref-to-llvm 2>&1 | FileCheck %s
// Since the error is at an unknown location, we use FileCheck instead of
// -veri-y-diagnostics here

// CHECK: conversion of memref memory space 1 : ui64 to integer address space failed. Consider adding memory space conversions.
// CHECK-LABEL: @invalid_int_conversion
func.func @invalid_int_conversion() {
     %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<10xf32, 1 : ui64>
    return
}
