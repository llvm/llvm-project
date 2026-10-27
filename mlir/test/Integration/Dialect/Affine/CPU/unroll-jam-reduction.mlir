// RUN: mlir-opt %s \
// RUN:   -pass-pipeline="builtin.module(func.func(affine-loop-unroll-jam{unroll-jam-factor=2}))" \
// RUN: | mlir-opt \
// RUN:   -lower-affine -convert-scf-to-cf \
// RUN:   -convert-index-to-llvm=index-bitwidth=64 -convert-arith-to-llvm \
// RUN:   -convert-func-to-llvm -convert-cf-to-llvm \
// RUN:   -reconcile-unrealized-casts | \
// RUN: mlir-runner -e main -entry-point-result=i32 | FileCheck %s

func.func @sum(%upper_bound: index, %init: i32) -> i32 {
  %result = affine.for %i = 0 to %upper_bound
      iter_args(%acc = %init) -> (i32) {
    %one = arith.constant 1 : i32
    %next = arith.addi %acc, %one : i32
    affine.yield %next : i32
  }
  return %result : i32
}

func.func @main() -> i32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c5 = arith.constant 5 : i32

  %result0 = func.call @sum(%c0, %c5) : (index, i32) -> i32
  %result1 = func.call @sum(%c1, %c5) : (index, i32) -> i32
  %result2 = func.call @sum(%c2, %c5) : (index, i32) -> i32
  %result3 = func.call @sum(%c3, %c5) : (index, i32) -> i32

  %expected0 = arith.constant 5 : i32
  %expected1 = arith.constant 6 : i32
  %expected2 = arith.constant 7 : i32
  %expected3 = arith.constant 8 : i32
  %diff0 = arith.subi %result0, %expected0 : i32
  %diff1 = arith.subi %result1, %expected1 : i32
  %diff2 = arith.subi %result2, %expected2 : i32
  %diff3 = arith.subi %result3, %expected3 : i32
  %sum01 = arith.addi %diff0, %diff1 : i32
  %sum23 = arith.addi %diff2, %diff3 : i32
  %error = arith.addi %sum01, %sum23 : i32
  return %error : i32
}

// CHECK: {{^0$}}
