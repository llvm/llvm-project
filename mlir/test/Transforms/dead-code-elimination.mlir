// RUN: mlir-opt -dce -split-input-file %s

// CHECK-LABEL: func @simple_test(
//  CHECK-SAME:     %[[arg0:.*]]: i16)
//  CHECK-NEXT:   %[[c5:.*]] = arith.constant 5 : i16
//  CHECK-NEXT:   %[[add:.*]] = arith.addi %[[c5]], %[[arg0]]
//  CHECK-NEXT:   return %[[add]]
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

// Note: Graph regions are not supported and skipped.

// CHECK-LABEL: test.graph_region {
//  CHECK-NEXT:   arith.addi
//  CHECK-NEXT:   arith.constant 5 : i16
//  CHECK-NEXT:   "test.baz"
//  CHECK-NEXT: }
test.graph_region {
  %1 = arith.addi %0, %0 : i16
  %0 = arith.constant 5 : i16
  "test.baz"() : () -> i32
}

// -----

// CHECK-LABEL: dead_blocks()
//  CHECK-NEXT:   cf.br ^[[bb3:.*]]
//  CHECK-NEXT: ^[[bb3]]:
//  CHECK-NEXT:   return
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
