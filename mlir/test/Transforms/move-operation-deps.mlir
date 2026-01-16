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

// Do not move across basic blocks
func.func @no_move_across_basic_blocks() -> (index, index) {
  %0 = "unmoved_op"() : () -> (index)
  %1 = "before"() : () -> (index)
  cf.br ^bb0(%0 : index)
 ^bb0(%arg0 : index) :
  %2 = arith.addi %arg0, %arg0 {moved_op} : index
  return %1, %2 : index, index
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0 : !transform.any_op {transform.readonly}) {
    %op1 = transform.structured.match ops{["before"]} in %arg0
        : (!transform.any_op) -> !transform.any_op
    %op2 = transform.structured.match ops{["arith.addi"]} in %arg0
        : (!transform.any_op) -> !transform.any_op
    %v1 = transform.get_result %op2[0] : (!transform.any_op) -> !transform.any_value
    // expected-remark@+1{{unsupported case of moving definition of value before an insertion point in a different basic block}}
    transform.test.move_value_defns %v1 before %op1
        : (!transform.any_value), !transform.any_op
    transform.yield
  }
}

// -----

func.func @move_isolated_from_above(%arg0 : index) -> () {
  %1 = "before"() : () -> (index)
  %2 = arith.addi %arg0, %arg0 {moved0} : index
  %3 = arith.muli %2, %2 {moved1} : index
  return
}
// CHECK-LABEL: func @move_isolated_from_above(
//       CHECK:   %[[MOVED0:.+]] = arith.addi {{.*}} {moved0}
//       CHECK:   %[[MOVED1:.+]] = arith.muli %[[MOVED0]], %[[MOVED0]] {moved1}
//       CHECK:   %[[BEFORE:.+]] = "before"

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0 : !transform.any_op {transform.readonly}) {
    %op1 = transform.structured.match ops{["before"]} in %arg0
        : (!transform.any_op) -> !transform.any_op
    %op2 = transform.structured.match ops{["arith.muli"]} in %arg0
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
