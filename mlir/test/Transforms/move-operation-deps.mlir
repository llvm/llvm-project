// RUN: mlir-opt --allow-unregistered-dialect --transform-interpreter --split-input-file --verify-diagnostics %s | FileCheck %s

// Check simple move of dependent operation before insertion.
func.func @simple_move(%arg0 : f32) -> f32 {
  %0 = "before"() : () -> (f32)
  %1 = arith.addf %arg0, %arg0 {moved_op} : f32
  %2 = "foo"(%1) : (f32) -> (f32)
  return %2 : f32
}
// CHECK-LABEL: func @simple_move(
//       CHECK:   %[[MOVED:.+]] = arith.addf {{.*}} {moved_op}
//       CHECK:   %[[BEFORE:.+]] = "before"
//       CHECK:   %[[FOO:.+]] = "foo"(%[[MOVED]])
//       CHECK:   return %[[FOO]]

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0 : !transform.any_op {transform.readonly}) {
    %op1 = transform.structured.match ops{["foo"]} in %arg0
        : (!transform.any_op) -> !transform.any_op
    %op2 = transform.structured.match ops{["before"]} in %arg0
        : (!transform.any_op) -> !transform.any_op
    transform.test.move_operand_deps %op1 before %op2
        : !transform.any_op, !transform.any_op
    transform.yield
  }
}

// -----

// Move operands that are implicitly captured by the op
func.func @move_region_dependencies(%arg0 : f32) -> f32 {
  %0 = "before"() : () -> (f32)
  %1 = arith.addf %arg0, %arg0 {moved_op} : f32
  %2 = "foo"() ({
    %3 = arith.mulf %1, %1 : f32
    "yield"(%3) : (f32) -> ()
  }) : () -> (f32)
  return %2 : f32
}
// CHECK-LABEL: func @move_region_dependencies(
//       CHECK:   %[[MOVED:.+]] = arith.addf {{.*}} {moved_op}
//       CHECK:   %[[BEFORE:.+]] = "before"
//       CHECK:   %[[FOO:.+]] = "foo"
//       CHECK:   return %[[FOO]]

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0 : !transform.any_op {transform.readonly}) {
    %op1 = transform.structured.match ops{["foo"]} in %arg0
        : (!transform.any_op) -> !transform.any_op
    %op2 = transform.structured.match ops{["before"]} in %arg0
        : (!transform.any_op) -> !transform.any_op
    transform.test.move_operand_deps %op1 before %op2
        : !transform.any_op, !transform.any_op
    transform.yield
  }
}

// -----

// Move operations in toplogical sort order
func.func @move_in_topological_sort_order(%arg0 : f32, %arg1 : f32) -> f32 {
  %0 = "before"() : () -> (f32)
  %1 = arith.addf %arg0, %arg0 {moved_op_1} : f32
  %2 = arith.addf %arg1, %arg1 {moved_op_2} : f32
  %3 = arith.mulf %1, %1 {moved_op_3} : f32
  %4 = arith.mulf %1, %3 {moved_op_4} : f32
  %5 = arith.mulf %2, %2 {moved_op_5} : f32
  %6 = "foo"(%4, %5) : (f32, f32) -> (f32)
  return %6 : f32
}
// CHECK-LABEL: func @move_in_topological_sort_order(
//       CHECK:   %[[MOVED_1:.+]] = arith.addf {{.*}} {moved_op_1}
//   CHECK-DAG:   %[[MOVED_2:.+]] = arith.mulf %[[MOVED_1]], %[[MOVED_1]] {moved_op_3}
//   CHECK-DAG:   %[[MOVED_3:.+]] = arith.mulf %[[MOVED_1]], %[[MOVED_2]] {moved_op_4}
//   CHECK-DAG:   %[[MOVED_4:.+]] = arith.addf {{.*}} {moved_op_2}
//   CHECK-DAG:   %[[MOVED_5:.+]] = arith.mulf %[[MOVED_4]], %[[MOVED_4]] {moved_op_5}
//       CHECK:   %[[BEFORE:.+]] = "before"
//       CHECK:   %[[FOO:.+]] = "foo"(%[[MOVED_3]], %[[MOVED_5]])
//       CHECK:   return %[[FOO]]

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0 : !transform.any_op {transform.readonly}) {
    %op1 = transform.structured.match ops{["foo"]} in %arg0
        : (!transform.any_op) -> !transform.any_op
    %op2 = transform.structured.match ops{["before"]} in %arg0
        : (!transform.any_op) -> !transform.any_op
    transform.test.move_operand_deps %op1 before %op2
        : !transform.any_op, !transform.any_op
    transform.yield
  }
}

// -----

func.func @move_region_dependencies(%arg0 : f32) -> f32 {
  %cond = arith.constant true
  %0 = "before"() : () -> (f32)
  %1 = arith.addf %arg0, %arg0 {moved_op_1} : f32
  %2 = scf.if %cond -> f32 {
    %inner = arith.mulf %1, %1 : f32
    scf.yield %inner : f32
  } else {
    scf.yield %1 : f32
  }
  %3 = "foo"() ({
    "yield"(%2) : (f32) -> ()
  }) : () -> (f32)
  return %3 : f32
}
// CHECK-LABEL: func @move_region_dependencies(
//       CHECK:   arith.constant true
//       CHECK:   %[[MOVED1:.+]] = arith.addf {{.*}} {moved_op_1}
//       CHECK:   %[[MOVED2:.+]] = scf.if
//       CHECK:   "before"
//       CHECK:   %[[FOO:.+]] = "foo"
//       CHECK:   return %[[FOO]]

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0 : !transform.any_op {transform.readonly}) {
    %op1 = transform.structured.match ops{["foo"]} in %arg0
        : (!transform.any_op) -> !transform.any_op
    %op2 = transform.structured.match ops{["before"]} in %arg0
        : (!transform.any_op) -> !transform.any_op
    transform.test.move_operand_deps %op1 before %op2
        : !transform.any_op, !transform.any_op
    transform.yield
  }
}

// -----

// Current implementation omits following basic block argument when
// computing slices. Verify that this gives expected result.
func.func @ignore_basic_block_arguments() -> f32 {
^bb0():
  %8 = "test"() : () -> (f32)
  return %8: f32
^bb1(%bbArg : f32):
  %cond = arith.constant true
  %0 = "before"() : () -> (f32)
  %1 = scf.if %cond -> f32 {
    %inner = arith.addf %bbArg, %bbArg : f32
    scf.yield %inner : f32
  } else {
    scf.yield %bbArg : f32
  }
  %2 = "foo"() ({
    "yield"(%1) : (f32) -> ()
  }) : () -> (f32)
  return %2 : f32
}
// CHECK-LABEL: func @ignore_basic_block_arguments()
//       CHECK:   arith.constant true
//       CHECK:   %[[MOVED:.+]] = scf.if
//       CHECK:   "before"
//       CHECK:   %[[FOO:.+]] = "foo"
//       CHECK:   return %[[FOO]]

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0 : !transform.any_op {transform.readonly}) {
    %op1 = transform.structured.match ops{["foo"]} in %arg0
        : (!transform.any_op) -> !transform.any_op
    %op2 = transform.structured.match ops{["before"]} in %arg0
        : (!transform.any_op) -> !transform.any_op
    transform.test.move_operand_deps %op1 before %op2
        : !transform.any_op, !transform.any_op
    transform.yield
  }
}

// -----

// Fail when the "before" operation is part of the operation slice.
func.func @do_not_move_slice() -> f32 {
  %0 = arith.constant {before} 1.0 : f32
  %1 = arith.addf %0, %0 {moved_op} : f32
  %2 = "foo"(%1) : (f32) -> (f32)
  return %2 : f32
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0 : !transform.any_op {transform.readonly}) {
    %op1 = transform.structured.match ops{["foo"]} in %arg0
        : (!transform.any_op) -> !transform.any_op
    %op2 = transform.structured.match attributes{before} in %arg0
        : (!transform.any_op) -> !transform.any_op
    // expected-remark@+1{{cannot move dependencies before operation in backward slice of op}}
    transform.test.move_operand_deps %op1 before %op2
        : !transform.any_op, !transform.any_op
    transform.yield
  }
}

// -----

// Fail when the "before" operation is part of the operation slice (computed
// when looking through implicitly captured values).
func.func @do_not_move_slice_region() -> f32 {
  %cond = arith.constant true
  %0 = arith.constant {before} 1.0 : f32
  %1 = scf.if %cond -> f32 {
    %inner = arith.addf %0, %0 : f32
    scf.yield %inner : f32
  } else {
    scf.yield %0 : f32
  }
  %2 = "foo"() ({
    "yield"(%1) : (f32) -> ()
  }) : () -> (f32)
  return %2 : f32
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0 : !transform.any_op {transform.readonly}) {
    %op1 = transform.structured.match ops{["foo"]} in %arg0
        : (!transform.any_op) -> !transform.any_op
    %op2 = transform.structured.match attributes{before} in %arg0
        : (!transform.any_op) -> !transform.any_op
    // expected-remark@+1{{cannot move dependencies before operation in backward slice of op}}
    transform.test.move_operand_deps %op1 before %op2
        : !transform.any_op, !transform.any_op
    transform.yield
  }
}

// -----

// Dont move ops when insertion point does not dominate the op
func.func @do_not_move(%arg0 : f32) -> f32 {
  %1 = arith.addf %arg0, %arg0 {moved_op} : f32
  %2 = "foo"() ({
    "yield"(%1) : (f32) -> ()
  }) : () -> (f32)
  %3 = "before"() : () -> f32
  return %2 : f32
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0 : !transform.any_op {transform.readonly}) {
    %op1 = transform.structured.match ops{["foo"]} in %arg0
        : (!transform.any_op) -> !transform.any_op
    %op2 = transform.structured.match ops{["before"]} in %arg0
        : (!transform.any_op) -> !transform.any_op
    // expected-remark@+1{{insertion point does not dominate op}}
    transform.test.move_operand_deps %op1 before %op2
        : !transform.any_op, !transform.any_op
    transform.yield
  }
}

// -----

// Check simple move value definitions before insertion operation.
func.func @simple_move_values(%arg0 : index) -> index {
  %c0 = arith.constant 0 : index
  %0 = "before"() : () -> (index)
  %1 = arith.addi %arg0, %c0 {"moved_op_1"} : index
  %2 = arith.subi %arg0, %c0 {"moved_op_2"} : index
  %3 = "foo"(%1, %2) : (index, index) -> (index)
  return %3 : index
}
// CHECK-LABEL: func @simple_move_values(
//       CHECK:   %[[MOVED1:.+]] = arith.addi {{.*}} {moved_op_1}
//       CHECK:   %[[MOVED2:.+]] = arith.subi {{.*}} {moved_op_2}
//       CHECK:   %[[BEFORE:.+]] = "before"
//       CHECK:   %[[FOO:.+]] = "foo"(%[[MOVED1]], %[[MOVED2]])
//       CHECK:   return %[[FOO]]

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0 : !transform.any_op {transform.readonly}) {
    %op1 = transform.structured.match ops{["arith.addi"]} in %arg0
        : (!transform.any_op) -> !transform.any_op
    %op2 = transform.structured.match ops{["arith.subi"]} in %arg0
        : (!transform.any_op) -> !transform.any_op
    %op3 = transform.structured.match ops{["before"]} in %arg0
        : (!transform.any_op) -> !transform.any_op
    %v1 = transform.get_result %op1[0] : (!transform.any_op) -> !transform.any_value
    %v2 = transform.get_result %op2[0] : (!transform.any_op) -> !transform.any_value
    transform.test.move_value_defns %v1, %v2 before %op3
        : (!transform.any_value, !transform.any_value), !transform.any_op
    transform.yield
  }
}

// -----

// Compute slice including the implicitly captured values.
func.func @move_region_dependencies_values(%arg0 : index, %cond : i1) -> index {
  %0 = "before"() : () -> (index)
  %1 = arith.addi %arg0, %arg0 {moved_op_1} : index
  %2 = scf.if %cond -> index {
    %3 = arith.muli %1, %1 {inner_op} : index
    scf.yield %3 : index
  } else {
    scf.yield %1 : index
  }
  return %2 : index
}
// CHECK-LABEL: func @move_region_dependencies_values(
//       CHECK:   %[[MOVED1:.+]] = arith.addi {{.*}} {moved_op_1}
//       CHECK:   scf.if
//       CHECK:     arith.muli %[[MOVED1]], %[[MOVED1]] {inner_op}
//       CHECK:   %[[BEFORE:.+]] = "before"

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0 : !transform.any_op {transform.readonly}) {
    %op1 = transform.structured.match ops{["scf.if"]} in %arg0
        : (!transform.any_op) -> !transform.any_op
    %op2 = transform.structured.match ops{["before"]} in %arg0
        : (!transform.any_op) -> !transform.any_op
    %v1 = transform.get_result %op1[0] : (!transform.any_op) -> !transform.any_value
    transform.test.move_value_defns %v1 before %op2
        : (!transform.any_value), !transform.any_op
    transform.yield
  }
}

// -----

// Move operations in toplogical sort order
func.func @move_values_in_topological_sort_order(%arg0 : index, %arg1 : index) -> index {
  %0 = "before"() : () -> (index)
  %1 = arith.addi %arg0, %arg0 {moved_op_1} : index
  %2 = arith.addi %arg1, %arg1 {moved_op_2} : index
  %3 = arith.muli %1, %1 {moved_op_3} : index
  %4 = arith.andi %1, %3 {moved_op_4} : index
  %5 = arith.subi %2, %2 {moved_op_5} : index
  %6 = "foo"(%4, %5) : (index, index) -> (index)
  return %6 : index
}
// CHECK-LABEL: func @move_values_in_topological_sort_order(
//       CHECK:   %[[MOVED_1:.+]] = arith.addi {{.*}} {moved_op_1}
//   CHECK-DAG:   %[[MOVED_2:.+]] = arith.muli %[[MOVED_1]], %[[MOVED_1]] {moved_op_3}
//   CHECK-DAG:   %[[MOVED_3:.+]] = arith.andi %[[MOVED_1]], %[[MOVED_2]] {moved_op_4}
//   CHECK-DAG:   %[[MOVED_4:.+]] = arith.addi {{.*}} {moved_op_2}
//   CHECK-DAG:   %[[MOVED_5:.+]] = arith.subi %[[MOVED_4]], %[[MOVED_4]] {moved_op_5}
//       CHECK:   %[[BEFORE:.+]] = "before"
//       CHECK:   %[[FOO:.+]] = "foo"(%[[MOVED_3]], %[[MOVED_5]])
//       CHECK:   return %[[FOO]]

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0 : !transform.any_op {transform.readonly}) {
    %op1 = transform.structured.match ops{["arith.andi"]} in %arg0
        : (!transform.any_op) -> !transform.any_op
    %op2 = transform.structured.match ops{["arith.subi"]} in %arg0
        : (!transform.any_op) -> !transform.any_op
    %op3 = transform.structured.match ops{["before"]} in %arg0
        : (!transform.any_op) -> !transform.any_op
    %v1 = transform.get_result %op1[0] : (!transform.any_op) -> !transform.any_value
    %v2 = transform.get_result %op2[0] : (!transform.any_op) -> !transform.any_value
    transform.test.move_value_defns %v1, %v2 before %op3
        : (!transform.any_value, !transform.any_value), !transform.any_op
    transform.yield
  }
}

// -----

// Move only those value definitions that are not dominated by insertion point

func.func @move_only_required_defns(%arg0 : index) -> (index, index, index, index) {
  %0 = "unmoved_op"() : () -> (index)
  %1 = "dummy_op"() : () -> (index)
  %2 = "before"() : () -> (index)
  %3 = arith.addi %arg0, %arg0 {moved_op} : index
  return %0, %1, %2, %3 : index, index, index, index
}
// CHECK-LABEL: func @move_only_required_defns(
//       CHECK:   %[[UNMOVED:.+]] = "unmoved_op"
//       CHECK:   %[[DUMMY:.+]] = "dummy_op"
//       CHECK:   %[[MOVED:.+]] = arith.addi {{.*}} {moved_op}
//       CHECK:   %[[BEFORE:.+]] = "before"

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0 : !transform.any_op {transform.readonly}) {
    %op1 = transform.structured.match ops{["unmoved_op"]} in %arg0
        : (!transform.any_op) -> !transform.any_op
    %op2 = transform.structured.match ops{["dummy_op"]} in %arg0
        : (!transform.any_op) -> !transform.any_op
    %op3 = transform.structured.match ops{["before"]} in %arg0
        : (!transform.any_op) -> !transform.any_op
    %op4 = transform.structured.match ops{["arith.addi"]} in %arg0
        : (!transform.any_op) -> !transform.any_op
    %v1 = transform.get_result %op1[0] : (!transform.any_op) -> !transform.any_value
    %v2 = transform.get_result %op4[0] : (!transform.any_op) -> !transform.any_value
    transform.test.move_value_defns %v1, %v2 before %op3
        : (!transform.any_value, !transform.any_value), !transform.any_op
    transform.yield
  }
}

// -----

// Move only those value definitions that are not dominated by insertion point (duplicate test)

func.func @move_only_required_defns_2(%arg0 : index) -> (index, index, index, index) {
  %0 = "unmoved_op"() : () -> (index)
  %1 = "dummy_op"() : () -> (index)
  %2 = "before"() : () -> (index)
  %3 = arith.subi %arg0, %arg0 {moved_op} : index
  return %0, %1, %2, %3 : index, index, index, index
}
// CHECK-LABEL: func @move_only_required_defns_2(
//       CHECK:   %[[UNMOVED:.+]] = "unmoved_op"
//       CHECK:   %[[DUMMY:.+]] = "dummy_op"
//       CHECK:   %[[MOVED:.+]] = arith.subi {{.*}} {moved_op}
//       CHECK:   %[[BEFORE:.+]] = "before"

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0 : !transform.any_op {transform.readonly}) {
    %op1 = transform.structured.match ops{["unmoved_op"]} in %arg0
        : (!transform.any_op) -> !transform.any_op
    %op2 = transform.structured.match ops{["dummy_op"]} in %arg0
        : (!transform.any_op) -> !transform.any_op
    %op3 = transform.structured.match ops{["before"]} in %arg0
        : (!transform.any_op) -> !transform.any_op
    %op4 = transform.structured.match ops{["arith.subi"]} in %arg0
        : (!transform.any_op) -> !transform.any_op
    %v1 = transform.get_result %op1[0] : (!transform.any_op) -> !transform.any_value
    %v2 = transform.get_result %op4[0] : (!transform.any_op) -> !transform.any_value
    transform.test.move_value_defns %v1, %v2 before %op3
        : (!transform.any_value, !transform.any_value), !transform.any_op
    transform.yield
  }
}

// -----

// Check handling of block arguments
func.func @move_with_block_arguments() -> (index, index) {
  %0 = "unmoved_op"() : () -> (index)
  cf.br ^bb0(%0 : index)
 ^bb0(%arg0 : index) :
  %1 = "before"() : () -> (index)
  %2 = arith.addi %arg0, %arg0 {moved_op} : index
  return %1, %2 : index, index
}
// CHECK-LABEL: func @move_with_block_arguments()
//       CHECK:   %[[MOVED:.+]] = arith.addi {{.*}} {moved_op}
//       CHECK:   %[[BEFORE:.+]] = "before"

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0 : !transform.any_op {transform.readonly}) {
    %op1 = transform.structured.match ops{["before"]} in %arg0
        : (!transform.any_op) -> !transform.any_op
    %op2 = transform.structured.match ops{["arith.addi"]} in %arg0
        : (!transform.any_op) -> !transform.any_op
    %v1 = transform.get_result %op2[0] : (!transform.any_op) -> !transform.any_value
    transform.test.move_value_defns %v1 before %op1
        : (!transform.any_value), !transform.any_op
    transform.yield
  }
}

// -----

// Successfully move operation between blocks in the same region.
// The function argument %arg0 dominates all blocks, so the move is valid.
func.func @move_between_blocks_same_region(%arg0 : index, %cond : i1) -> index {
  %0 = "before"() : () -> (index)
  cf.cond_br %cond, ^bb1, ^bb2(%arg0 : index)
^bb1:
  %1 = arith.addi %arg0, %arg0 {to_move} : index
  cf.br ^bb2(%1 : index)
^bb2(%result : index):
  return %result : index
}
// CHECK-LABEL: func @move_between_blocks_same_region
// CHECK:         %[[MOVED:.+]] = arith.addi {{.*}} {to_move}
// CHECK:         "before"

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0 : !transform.any_op {transform.readonly}) {
    %op1 = transform.structured.match ops{["before"]} in %arg0
        : (!transform.any_op) -> !transform.any_op
    %op2 = transform.structured.match attributes{to_move} in %arg0
        : (!transform.any_op) -> !transform.any_op
    %v1 = transform.get_result %op2[0] : (!transform.any_op) -> !transform.any_value
    transform.test.move_value_defns %v1 before %op1
        : (!transform.any_value), !transform.any_op
    transform.yield
  }
}

// -----


//===----------------------------------------------------------------------===//
// Cross-region move tests
//===----------------------------------------------------------------------===//

// Move multiple values with dependencies out of nested region
func.func @move_chain_out_of_region(%arg0 : index, %cond : i1) -> index {
  %0 = "before"() : () -> (index)
  %1 = scf.if %cond -> index {
    %2 = arith.addi %arg0, %arg0 {dep1} : index
    %3 = arith.muli %2, %2 {dep2} : index
    %4 = arith.subi %3, %arg0 {to_move} : index
    scf.yield %4 : index
  } else {
    scf.yield %arg0 : index
  }
  return %1 : index
}
// CHECK-LABEL: func @move_chain_out_of_region(
//       CHECK:   arith.addi {{.*}} {dep1}
//       CHECK:   arith.muli {{.*}} {dep2}
//       CHECK:   arith.subi {{.*}} {to_move}
//       CHECK:   "before"
//       CHECK:   scf.if

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0 : !transform.any_op {transform.readonly}) {
    %op1 = transform.structured.match ops{["before"]} in %arg0
        : (!transform.any_op) -> !transform.any_op
    %op2 = transform.structured.match attributes{to_move} in %arg0
        : (!transform.any_op) -> !transform.any_op
    %v1 = transform.get_result %op2[0] : (!transform.any_op) -> !transform.any_value
    transform.test.move_value_defns %v1 before %op1
        : (!transform.any_value), !transform.any_op
    transform.yield
  }
}

// -----

// moveValueDefinitions: cannot move when slice has side-effecting ops
func.func @move_value_defs_side_effecting(%arg0 : memref<index>) -> index {
  %0 = "before"() : () -> (index)
  %1 = memref.load %arg0[] : memref<index>
  %2 = arith.muli %1, %1 {moved_op} : index
  return %2 : index
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0 : !transform.any_op {transform.readonly}) {
    %op1 = transform.structured.match ops{["before"]} in %arg0
        : (!transform.any_op) -> !transform.any_op
    %op2 = transform.structured.match attributes{moved_op} in %arg0
        : (!transform.any_op) -> !transform.any_op
    %v1 = transform.get_result %op2[0] : (!transform.any_op) -> !transform.any_value
    // expected-remark@+1{{cannot move value definitions with side-effecting operations in the slice}}
    transform.test.move_value_defns %v1 before %op1
        : (!transform.any_value), !transform.any_op
    transform.yield
  }
}

// -----

// moveOperationDependencies: cannot move when slice has side-effecting ops
func.func @move_op_deps_side_effecting(%arg0 : memref<index>) -> index {
  %0 = "before"() : () -> (index)
  %1 = memref.load %arg0[] : memref<index>
  %2 = arith.muli %1, %1 {moved_op} : index
  %3 = "foo"(%2) : (index) -> (index)
  return %3 : index
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0 : !transform.any_op {transform.readonly}) {
    %op1 = transform.structured.match ops{["foo"]} in %arg0
        : (!transform.any_op) -> !transform.any_op
    %op2 = transform.structured.match ops{["before"]} in %arg0
        : (!transform.any_op) -> !transform.any_op
    // expected-remark@+1{{cannot move operation with side-effecting dependencies}}
    transform.test.move_operand_deps %op1 before %op2
        : !transform.any_op, !transform.any_op
    transform.yield
  }
}

// -----

// Can move op using outer loop's IV when staying within outer loop
func.func @move_op_using_outer_loop_iv(%lb : index, %ub : index, %step : index) -> index {
  %result = scf.for %outer_iv = %lb to %ub step %step iter_args(%acc = %lb) -> index {
    %before = "before"() : () -> (index)
    scf.for %inner_iv = %lb to %ub step %step {
      // Uses outer_iv which dominates within the outer loop body
      %x = arith.addi %outer_iv, %outer_iv {to_move} : index
      "use"(%x) : (index) -> ()
    }
    scf.yield %acc : index
  }
  return %result : index
}
// CHECK-LABEL: func @move_op_using_outer_loop_iv(
//       CHECK:   scf.for %[[OUTER_IV:[a-zA-Z0-9_]+]] =
//       CHECK:     arith.addi %[[OUTER_IV]], %[[OUTER_IV]] {to_move}
//       CHECK:     "before"
//       CHECK:     scf.for

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0 : !transform.any_op {transform.readonly}) {
    %op1 = transform.structured.match ops{["before"]} in %arg0
        : (!transform.any_op) -> !transform.any_op
    %op2 = transform.structured.match attributes{to_move} in %arg0
        : (!transform.any_op) -> !transform.any_op
    %v1 = transform.get_result %op2[0] : (!transform.any_op) -> !transform.any_value
    transform.test.move_value_defns %v1 before %op1
        : (!transform.any_value), !transform.any_op
    transform.yield
  }
}

// -----

// Move out of doubly nested non-isolated region
func.func @move_out_of_doubly_nested_region(%arg0 : index, %cond1 : i1, %cond2 : i1) -> index {
  %0 = "before"() : () -> (index)
  %1 = scf.if %cond1 -> index {
    %2 = scf.if %cond2 -> index {
      %3 = arith.addi %arg0, %arg0 {to_move} : index
      scf.yield %3 : index
    } else {
      scf.yield %arg0 : index
    }
    scf.yield %2 : index
  } else {
    scf.yield %arg0 : index
  }
  return %1 : index
}
// CHECK-LABEL: func @move_out_of_doubly_nested_region(
//       CHECK:   %[[MOVED:.+]] = arith.addi {{.*}} {to_move}
//       CHECK:   %[[BEFORE:.+]] = "before"
//       CHECK:   scf.if
//       CHECK:     scf.if
//       CHECK:       scf.yield %[[MOVED]]

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0 : !transform.any_op {transform.readonly}) {
    %op1 = transform.structured.match ops{["before"]} in %arg0
        : (!transform.any_op) -> !transform.any_op
    %op2 = transform.structured.match attributes{to_move} in %arg0
        : (!transform.any_op) -> !transform.any_op
    %v1 = transform.get_result %op2[0] : (!transform.any_op) -> !transform.any_value
    transform.test.move_value_defns %v1 before %op1
        : (!transform.any_value), !transform.any_op
    transform.yield
  }
}

// -----

// Move operand deps out of nested region
func.func @move_operand_deps_out_of_region(%arg0 : index, %cond : i1) -> index {
  %0 = "before"() : () -> (index)
  %1 = scf.if %cond -> index {
    %2 = arith.addi %arg0, %arg0 {dep} : index
    %3 = "foo"(%2) {target} : (index) -> (index)
    scf.yield %3 : index
  } else {
    scf.yield %arg0 : index
  }
  return %1 : index
}
// CHECK-LABEL: func @move_operand_deps_out_of_region(
//       CHECK:   %[[DEP:.+]] = arith.addi {{.*}} {dep}
//       CHECK:   %[[BEFORE:.+]] = "before"
//       CHECK:   scf.if
//       CHECK:     "foo"(%[[DEP]]) {target}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0 : !transform.any_op {transform.readonly}) {
    %op1 = transform.structured.match ops{["foo"]} in %arg0
        : (!transform.any_op) -> !transform.any_op
    %op2 = transform.structured.match ops{["before"]} in %arg0
        : (!transform.any_op) -> !transform.any_op
    transform.test.move_operand_deps %op1 before %op2
        : !transform.any_op, !transform.any_op
    transform.yield
  }
}

// -----

// Cannot move op that depends on loop induction variable (block argument)
func.func @cannot_move_op_using_loop_iv(%arg0 : index, %lb : index, %ub : index, %step : index) -> index {
  %0 = "before"() : () -> (index)
  %1 = scf.for %iv = %lb to %ub step %step iter_args(%acc = %arg0) -> index {
    %2 = arith.addi %iv, %iv {to_move} : index
    %3 = arith.addi %acc, %2 : index
    scf.yield %3 : index
  }
  return %1 : index
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0 : !transform.any_op {transform.readonly}) {
    %op1 = transform.structured.match ops{["before"]} in %arg0
        : (!transform.any_op) -> !transform.any_op
    %op2 = transform.structured.match attributes{to_move} in %arg0
        : (!transform.any_op) -> !transform.any_op
    %v1 = transform.get_result %op2[0] : (!transform.any_op) -> !transform.any_value
    // expected-remark@+1{{moving op would break dominance for block argument operand}}
    transform.test.move_value_defns %v1 before %op1
        : (!transform.any_value), !transform.any_op
    transform.yield
  }
}

// -----

// Cannot move out of an isolated-from-above region, even when op is in a
// non-isolated region nested inside the isolated region
func.func @cannot_move_out_of_isolated_region(%arg0 : index, %cond : i1) -> index {
  %0 = "before"() : () -> (index)
  %1 = "test.isolated_one_region_op"(%arg0, %cond) ({
    ^bb0(%inner_arg: index, %inner_cond: i1):
      // scf.if is NOT isolated, but it's inside an isolated region
      %2 = scf.if %inner_cond -> index {
        %3 = arith.addi %inner_arg, %inner_arg {to_move} : index
        scf.yield %3 : index
      } else {
        scf.yield %inner_arg : index
      }
      "test.region_yield"(%2) : (index) -> ()
  }) : (index, i1) -> (index)
  return %1 : index
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0 : !transform.any_op {transform.readonly}) {
    %op1 = transform.structured.match ops{["before"]} in %arg0
        : (!transform.any_op) -> !transform.any_op
    %op2 = transform.structured.match attributes{to_move} in %arg0
        : (!transform.any_op) -> !transform.any_op
    %v1 = transform.get_result %op2[0] : (!transform.any_op) -> !transform.any_value
    // expected-remark@+1{{cannot move value definition across isolated-from-above region}}
    transform.test.move_value_defns %v1 before %op1
        : (!transform.any_value), !transform.any_op
    transform.yield
  }
}

// -----

// Fail when trying to move an operation whose region captures a block argument
// that wouldn't dominate at the insertion point.
func.func @captured_block_arg_does_not_dominate(%arg0 : f32, %cond : i1) -> f32 {
  %0 = arith.addf %arg0, %arg0 {before} : f32
  cf.br ^bb1(%0 : f32)
^bb1(%bbArg : f32):
  // scf.if will be part of the slice that needs to move.
  // It has a region that captures %bbArg from bb1.
  // Moving it before the {before} op in the entry block would be invalid
  // because %bbArg (a block argument of bb1) doesn't dominate the entry block.
  %1 = scf.if %cond -> f32 {
    %inner = arith.addf %bbArg, %bbArg : f32
    scf.yield %inner : f32
  } else {
    scf.yield %bbArg : f32
  }
  %2 = arith.mulf %1, %1 {target} : f32
  return %2 : f32
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0 : !transform.any_op {transform.readonly}) {
    %op1 = transform.structured.match attributes{before} in %arg0
        : (!transform.any_op) -> !transform.any_op
    %op2 = transform.structured.match attributes{target} in %arg0
        : (!transform.any_op) -> !transform.any_op
    %v1 = transform.get_result %op2[0] : (!transform.any_op) -> !transform.any_value
    // expected-remark@+1{{moving op would break dominance for block argument operand}}
    transform.test.move_value_defns %v1 before %op1
        : (!transform.any_value), !transform.any_op
    transform.yield
  }
}
