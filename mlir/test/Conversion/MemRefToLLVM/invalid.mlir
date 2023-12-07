// RUN: mlir-opt %s -finalize-memref-to-llvm -split-input-file 2>&1 | FileCheck %s
// Since the error is at an unknown location, we use FileCheck instead of
// -veri-y-diagnostics here

// CHECK: conversion of memref memory space "foo" to integer address space failed. Consider adding memory space conversions.
// CHECK-LABEL: @bad_address_space
func.func @bad_address_space(%a: memref<2xindex, "foo">) {
    %c0 = arith.constant 0 : index
    // CHECK: memref.store
    memref.store %c0, %a[%c0] : memref<2xindex, "foo">
    return
}

// -----
