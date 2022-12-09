// RUN: mlir-opt %s -test-transform-dialect-interpreter -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL: @get_parent_for_op
func.func @get_parent_for_op(%arg0: index, %arg1: index, %arg2: index) {
  // expected-remark @below {{first loop}}
  affine.for %i = %arg0 to %arg1 {
    // expected-remark @below {{second loop}}
    affine.for %j = %arg0 to %arg1 {
      // expected-remark @below {{third loop}}
      affine.for %k = %arg0 to %arg1 {
        arith.addi %i, %j : index
      }
    }
  }
  return
}

transform.sequence failures(propagate) {
^bb1(%arg1: !pdl.operation):
  %0 = transform.structured.match ops{["arith.addi"]} in %arg1
  // CHECK: = transform.affine.get_parent_for
  %1 = transform.affine.get_parent_for %0 : (!pdl.operation) -> !transform.op<"affine.for">
  %2 = transform.affine.get_parent_for %0 { num_loops = 2 } : (!pdl.operation) -> !transform.op<"affine.for">
  %3 = transform.affine.get_parent_for %0 { num_loops = 3 } : (!pdl.operation) -> !transform.op<"affine.for">
  transform.test_print_remark_at_operand %1, "third loop" : !transform.op<"affine.for">
  transform.test_print_remark_at_operand %2, "second loop" : !transform.op<"affine.for">
  transform.test_print_remark_at_operand %3, "first loop" : !transform.op<"affine.for">
}

// -----

func.func @get_parent_for_op_no_loop(%arg0: index, %arg1: index) {
  // expected-note @below {{target op}}
  arith.addi %arg0, %arg1 : index
  return
}

transform.sequence failures(propagate) {
^bb1(%arg1: !pdl.operation):
  %0 = transform.structured.match ops{["arith.addi"]} in %arg1
  // expected-error @below {{could not find an 'affine.for' parent}}
  %1 = transform.affine.get_parent_for %0 : (!pdl.operation) -> !transform.op<"affine.for">
}

// -----

func.func @loop_unroll_op() {
  %c0 = arith.constant 0 : index
  %c42 = arith.constant 42 : index
  %c5 = arith.constant 5 : index
  // CHECK: affine.for %[[I:.+]] =
  // expected-remark @below {{affine for loop}}
  affine.for %i = %c0 to %c42 {
    // CHECK-COUNT-4: arith.addi
    arith.addi %i, %i : index
  }
  return
}

transform.sequence failures(propagate) {
^bb1(%arg1: !pdl.operation):
  %0 = transform.structured.match ops{["arith.addi"]} in %arg1
  %1 = transform.affine.get_parent_for %0 : (!pdl.operation) -> !transform.op<"affine.for">
  transform.test_print_remark_at_operand %1, "affine for loop" : !transform.op<"affine.for">
  transform.affine.unroll %1 { factor = 4 } : !transform.op<"affine.for">
}

