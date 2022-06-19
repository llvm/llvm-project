// RUN: mlir-opt --pass-pipeline="func.func(test-patterns)" %s | FileCheck %s

// CHECK-LABEL: func @test_reorder_constants_and_match
func.func @test_reorder_constants_and_match(%arg0 : i32) -> (i32) {
  // CHECK: %[[CST:.+]] = arith.constant 43
  %cst = arith.constant 43 : i32
  // CHECK: return %[[CST]]
  %y = "test.op_commutative2"(%cst, %arg0) : (i32, i32) -> i32
  %x = "test.op_commutative2"(%y, %arg0) : (i32, i32) -> i32
  return %x : i32
}
