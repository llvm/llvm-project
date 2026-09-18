// RUN: mlir-opt %s -pass-pipeline='builtin.module(func.func(affine-loop-tile{tile-size=8},affine-loop-fusion))' | FileCheck %s

// Ensure sibling fusion does not erase a result-producing loop with live uses.

// CHECK-LABEL: func.func @preserve_live_sibling_results
// CHECK-COUNT-2: = affine.for
// CHECK: arith.addi
// CHECK: return
func.func @preserve_live_sibling_results() -> i64 {
  %a = memref.alloc() : memref<4xi64>
  %b = memref.alloc() : memref<4xi64>
  %c = memref.alloc() : memref<4xi64>
  %c0 = arith.constant 0 : i64
  %c3 = arith.constant 3 : i64
  %c7 = arith.constant 7 : i64

  affine.for %i = 0 to 4 {
    affine.store %c3, %a[%i] : memref<4xi64>
  }
  affine.for %i = 0 to 4 {
    affine.store %c0, %b[%i] : memref<4xi64>
    affine.store %c0, %c[%i] : memref<4xi64>
  }
  affine.for %i = 0 to 4 {
    %v = affine.load %a[%i] : memref<4xi64>
    %p = arith.muli %v, %c3 : i64
    affine.store %p, %b[%i] : memref<4xi64>
  }
  affine.for %i = 0 to 4 {
    %v = affine.load %b[%i] : memref<4xi64>
    %q = arith.addi %v, %c7 : i64
    affine.store %q, %c[%i] : memref<4xi64>
  }

  %sum0 = affine.for %i = 0 to 4 iter_args(%sum = %c0) -> i64 {
    %v = affine.load %b[%i] : memref<4xi64>
    %next = arith.addi %sum, %v : i64
    affine.yield %next : i64
  }
  %sum1 = affine.for %i = 0 to 4 iter_args(%sum = %c0) -> i64 {
    %v = affine.load %c[%i] : memref<4xi64>
    %next = arith.addi %sum, %v : i64
    affine.yield %next : i64
  }

  %sum = arith.addi %sum0, %sum1 : i64
  return %sum : i64
}
