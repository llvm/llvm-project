// RUN: mlir-opt %s -finalize-memref-to-llvm 2>&1 | FileCheck %s
// Since the error is at an unknown location, we use FileCheck instead of
// -veri-y-diagnostics here

// CHECK: redefinition of reserved function 'malloc' of different type '!llvm.func<void (i64)>' is prohibited
llvm.func @malloc(i64)
func.func @redef_reserved() {
    %alloc = memref.alloc() : memref<1024x64xf32, 1>
    llvm.return
}

// CHECK: conversion of memref memory space "foo" to integer address space failed. Consider adding memory space conversions.
// CHECK-LABEL: @bad_address_space
func.func @bad_address_space(%a: memref<2xindex, "foo">) {
    %c0 = arith.constant 0 : index
    // CHECK: memref.store
    memref.store %c0, %a[%c0] : memref<2xindex, "foo">
    return
}
