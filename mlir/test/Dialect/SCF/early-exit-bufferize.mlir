// RUN: mlir-opt %s -one-shot-bufferize="allow-unknown-ops bufferize-function-boundaries" -split-input-file | FileCheck %s

// A break carries a token before its payload. Only yields contribute aliases
// and buffer types for an if result.
// CHECK-LABEL: func.func @break_then(
// CHECK-SAME: %[[COND:.*]]: i1, %[[INPUT:.*]]: memref<5xf32
// CHECK: scf.if %[[COND]] -> (memref<5xf32
// CHECK: scf.break
// CHECK: } else {
// CHECK: scf.yield %[[INPUT]] : memref<5xf32
func.func @break_then(%cond: i1, %input: tensor<5xf32>) {
  scf.loop token(%token) {
    %r = scf.if %cond -> tensor<5xf32> {
      scf.break [%token]
    } else {
      scf.yield %input : tensor<5xf32>
    }
    "test.use"(%r) : (tensor<5xf32>) -> ()
    scf.break [%token]
  }
  return
}

// -----

// CHECK-LABEL: func.func @break_else(
// CHECK-SAME: %[[COND:.*]]: i1, %[[INPUT:.*]]: memref<5xf32
// CHECK: scf.if %[[COND]] -> (memref<5xf32
// CHECK: scf.yield %[[INPUT]] : memref<5xf32
// CHECK: } else {
// CHECK: scf.break
func.func @break_else(%cond: i1, %input: tensor<5xf32>) {
  scf.loop token(%token) {
    %r = scf.if %cond -> tensor<5xf32> {
      scf.yield %input : tensor<5xf32>
    } else {
      scf.break [%token]
    }
    "test.use"(%r) : (tensor<5xf32>) -> ()
    scf.break [%token]
  }
  return
}
