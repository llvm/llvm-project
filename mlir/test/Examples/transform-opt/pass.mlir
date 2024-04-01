// RUN: mlir-transform-opt %s | FileCheck %s

module attributes {transform.with_named_sequence} {
  // CHECK-LABEL: @return_42
  // CHECK: %[[C42:.+]] = arith.constant 42
  // CHECK: return %[[C42]]
  func.func @return_42() -> i32 {
    %0 = arith.constant 21 : i32
    %1 = arith.constant 2 : i32
    %2 = arith.muli %0, %1 : i32
    return %2 : i32
  }

  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    %arg1 = transform.apply_registered_pass "canonicalize" to %arg0 : (!transform.any_op) -> !transform.any_op
    transform.print %arg1 : !transform.any_op
    transform.yield
  }
}
