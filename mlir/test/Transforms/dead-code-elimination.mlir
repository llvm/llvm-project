// RUN: mlir-opt -trivial-dce -split-input-file %s | FileCheck %s
// RUN: mlir-opt -pass-pipeline="builtin.module(func.func(trivial-dce{recursive=false}))" -split-input-file %s | FileCheck %s --check-prefix=NONREC
// RUN: mlir-opt -pass-pipeline="builtin.module(func.func(trivial-dce{remove-blocks=false}))" -split-input-file %s | FileCheck %s --check-prefix=NOBLOCKS

// CHECK-LABEL: func @simple_test(
//  CHECK-SAME:     %[[arg0:.*]]: i16)
//  CHECK-NEXT:   %[[c5:.*]] = arith.constant 5 : i16
//  CHECK-NEXT:   %[[add:.*]] = arith.addi %[[c5]], %[[arg0]]
//  CHECK-NEXT:   return %[[add]]
// NONREC-LABEL: func @simple_test(
//  NONREC-SAME:     %[[arg0:.*]]: i16)
//  NONREC-NEXT:   %[[c5:.*]] = arith.constant 5 : i16
//  NONREC-NEXT:   %[[add:.*]] = arith.addi %[[c5]], %[[arg0]]
//  NONREC-NEXT:   return %[[add]]
func.func @simple_test(%arg0: i16) -> i16 {
  %0 = arith.constant 5 : i16
  %1 = arith.addi %0, %arg0 : i16
  %2 = arith.addi %1, %1 : i16
  %3 = arith.addi %2, %1 : i16
  return %1 : i16
}

// -----

// CHECK-LABEL: func @eliminate_from_region
//  CHECK-NEXT:   scf.for {{.*}} {
//  CHECK-NEXT:     arith.constant
//  CHECK-NEXT:     "test.print"
//  CHECK-NEXT:   }
//  CHECK-NEXT:   return
// NONREC-LABEL: func @eliminate_from_region
//  NONREC:      scf.for {{.*}} {
//  NONREC-NEXT:   arith.constant 5
//  NONREC-NEXT:   arith.constant 10
//  NONREC-NEXT:   "test.print"
func.func @eliminate_from_region(%lb: index, %ub: index, %step: index) {
  scf.for %iv = %lb to %ub step %step {
    %0 = arith.constant 5 : i16
    %1 = arith.constant 10 : i16
    "test.print"(%0) : (i16) -> ()
  }
  return
}

// -----

// CHECK-LABEL: func @eliminate_op_with_region
//  CHECK-NEXT:   return
func.func @eliminate_op_with_region(%lb: index, %ub: index, %step: index) {
  %c0 = arith.constant 0 : i16
  %0 = scf.for %iv = %lb to %ub step %step iter_args(%iter = %c0) -> i16 {
    %0 = arith.constant 5 : i16
    %added = arith.addi %iter, %0 : i16
    scf.yield %added : i16
  }
  return
}

// -----

// CHECK-LABEL: func @unstructured_control_flow(
//  CHECK-SAME:     %[[arg0:.*]]: i16)
//  CHECK-NEXT:   %[[c5:.*]] = arith.constant 5 : i16
//  CHECK-NEXT:   cf.br ^[[bb2:.*]]
//  CHECK-NEXT: ^[[bb1:.*]]:  // pred
//  CHECK-NEXT:   cf.br ^[[bb3:.*]]
//  CHECK-NEXT: ^[[bb2]]:
//  CHECK-NEXT:   %[[add:.*]] = arith.addi %[[c5]], %[[arg0]]
//  CHECK-NEXT:   cf.br ^[[bb1]]
//  CHECK-NEXT: ^[[bb3]]:
//  CHECK-NEXT:   return %[[add]]
func.func @unstructured_control_flow(%arg0: i16) -> i16 {
  %0 = arith.constant 5 : i16
  cf.br ^bb2
^bb1:
  %3 = arith.addi %1, %1 : i16
  %4 = arith.addi %3, %2 : i16
  cf.br ^bb3
^bb2:
  %1 = arith.addi %0, %arg0 : i16
  %2 = arith.subi %0, %arg0 : i16
  cf.br ^bb1
^bb3:
  return %1 : i16
}

// -----

// CHECK-LABEL: func @remove_dead_block()
//  CHECK-NEXT:   cf.br ^[[bb2:.*]]
//  CHECK-NEXT: ^[[bb2]]:
//  CHECK-NEXT:   return
// NONREC-LABEL: func @remove_dead_block()
//  NONREC-NEXT:   cf.br ^[[bb2:.*]]
//  NONREC-NEXT: ^[[bb2]]:
//  NONREC-NEXT:   return
// NOBLOCKS-LABEL: func @remove_dead_block()
//  NOBLOCKS-NEXT:   cf.br ^[[bb2:.*]]
//  NOBLOCKS-NEXT: ^[[bb1:.*]]:  // no predecessors
//  NOBLOCKS-NEXT:   cf.br ^[[bb2]]
//  NOBLOCKS-NEXT: ^[[bb2]]:
//  NOBLOCKS-NEXT:   return
func.func @remove_dead_block() {
  cf.br ^bb2
^bb1:
  %0 = arith.constant 0 : i16
  cf.br ^bb2
^bb2:
  return
}

// -----

// CHECK-LABEL: func @potentially_side_effecting_op()
//  CHECK-NEXT:   "test.print"
//  CHECK-NEXT:   return
func.func @potentially_side_effecting_op() {
  "test.print"() : () -> ()
  return
}

// -----

// CHECK-LABEL: func @miss_nested_operand_of_erased_op(
//  CHECK-NOT:    scf.if
//  CHECK-NOT:    arith.addi
//  CHECK:        return
func.func @miss_nested_operand_of_erased_op(%arg0: i16, %cond: i1) {
  cf.br ^producer

^user:
  %dead_user = arith.addi %if_result, %if_result : i16
  cf.br ^exit

^if_block:
  %if_result = scf.if %cond -> i16 {
    %nested_use = arith.addi %producer_value, %producer_value : i16
    scf.yield %nested_use : i16
  } else {
    scf.yield %producer_value : i16
  }
  cf.br ^user

^producer:
  %producer_value = arith.addi %arg0, %arg0 : i16
  cf.br ^if_block

^exit:
  return
}

// -----

// CHECK-LABEL: test.graph_region {
//  CHECK-NEXT:   "test.baz"
//  CHECK-NEXT: }
test.graph_region {
  %1 = arith.addi %0, %0 : i16
  %0 = arith.constant 5 : i16
  "test.baz"() : () -> i32
}

// -----

// CHECK-LABEL: test.graph_region attributes {cycle} {
//  CHECK-NEXT:   %[[a:.*]] = arith.addi %[[b:.*]], %[[b]] : i16
//  CHECK-NEXT:   %[[b]] = arith.addi %[[a]], %[[a]] : i16
//  CHECK-NEXT:   "test.baz"
//  CHECK-NEXT: }
test.graph_region attributes {cycle} {
  %0 = arith.addi %1, %1 : i16
  %1 = arith.addi %0, %0 : i16
  "test.baz"() : () -> i32
}

// -----

// CHECK-LABEL: dead_blocks()
//  CHECK-NEXT:   cf.br ^[[bb3:.*]]
//  CHECK-NEXT: ^[[bb3]]:
//  CHECK-NEXT:   return
// NOBLOCKS-LABEL: dead_blocks()
//  NOBLOCKS-NEXT:   cf.br ^[[bb3:.*]]
//  NOBLOCKS-NEXT: ^[[bb1:.*]]:  // pred
//  NOBLOCKS-NEXT:   "test.print"
//  NOBLOCKS-NEXT:   cf.br ^[[bb2:.*]]
//  NOBLOCKS-NEXT: ^[[bb2]]:  // pred
//  NOBLOCKS-NEXT:   cf.br ^[[bb1]]
//  NOBLOCKS-NEXT: ^[[bb3]]:
//  NOBLOCKS-NEXT:   return
func.func @dead_blocks() {
  cf.br ^bb3
^bb1:
  "test.print"() : () -> ()
  cf.br ^bb2
^bb2:
  cf.br ^bb1
^bb3:
  return
}
