// RUN: mlir-opt %s -test-legalize-type-conversion="allow-pattern-rollback=0" -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL: func @simple_context_aware_conversion_1()
func.func @simple_context_aware_conversion_1() attributes {increment = 1 : i64} {
  // Case 1: Convert i37 --> i38.
  // CHECK: %[[cast:.*]] = builtin.unrealized_conversion_cast %{{.*}} : i37 to i38
  // CHECK: "test.legal_op_d"(%[[cast]]) : (i38) -> ()
  %0 = "test.context_op"() : () -> (i37)
  "test.replace_with_legal_op"(%0) : (i37) -> ()
  return
}

// CHECK-LABEL: func @simple_context_aware_conversion_2()
func.func @simple_context_aware_conversion_2() attributes {increment = 2 : i64} {
  // Case 2: Convert i37 --> i39.
  // CHECK: %[[cast:.*]] = builtin.unrealized_conversion_cast %{{.*}} : i37 to i39
  // CHECK: "test.legal_op_d"(%[[cast]]) : (i39) -> ()
  %0 = "test.context_op"() : () -> (i37)
  "test.replace_with_legal_op"(%0) : (i37) -> ()
  return
}

// -----

// Note: This test case does not work with allow-pattern-rollback=1. When
// rollback is enabled, the type converter cannot find the enclosing function
// because the operand of the scf.yield is pointing to a detached block.

// CHECK-LABEL: func @convert_block_arguments
//       CHECK:   %[[cast:.*]] = builtin.unrealized_conversion_cast %{{.*}} : i37 to i38
//       CHECK:   scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[iter:.*]] = %[[cast]]) -> (i38) {
//       CHECK:     scf.yield %[[iter]] : i38
//       CHECK:   }
func.func @convert_block_arguments(%lb: index, %ub: index, %step: index) attributes {increment = 1 : i64} {
  %0 = "test.context_op"() : () -> (i37)
  scf.for %iv = %lb to %ub step %step iter_args(%arg0 = %0) -> i37 {
    scf.yield %arg0 : i37
  }
  return
}
