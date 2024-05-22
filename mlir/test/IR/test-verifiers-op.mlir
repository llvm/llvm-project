// RUN: mlir-opt %s | FileCheck %s

// CHECK-LABEL: func @no_overflow_on_test_verifiers_op
func.func @no_overflow_on_test_verifiers_op() {
  %0 = arith.constant 1 : i32
  "test.verifiers"(%0) ({
    %1 = arith.constant 2 : i32
    "test.verifiers"(%1) ({
      %2 = arith.constant 3 : index
    }) : (i32) -> ()
  }) : (i32) -> ()
  return
}
