// RUN: mlir-opt -allow-unregistered-dialect -split-input-file -test-pattern-selective-replacement -verify-diagnostics %s | FileCheck %s

// Test that operations can be selectively replaced.

// CHECK-LABEL: @test1
// CHECK-SAME: %[[ARG0:.*]]: i32, %[[ARG1:.*]]: i32
func.func @test1(%arg0: i32, %arg1 : i32) -> () {
  // CHECK: arith.addi %[[ARG1]], %[[ARG1]]
  // CHECK-NEXT: "test.return"(%[[ARG0]]
  %cast = "test.cast"(%arg0, %arg1) : (i32, i32) -> (i32)
  %non_terminator = arith.addi %cast, %cast : i32
  "test.return"(%cast, %non_terminator) : (i32, i32) -> ()
}

// -----
