// RUN: mlir-opt --allow-unregistered-dialect --transform-interpreter --split-input-file --verify-diagnostics %s | FileCheck %s

// Check simple move of dependent operation before insertion.
func.func @simple_move() -> f32 {
  %0 = "before"() : () -> (f32)
  %1 = "moved_op"() : () -> (f32)
  %2 = "foo"(%1) : (f32) -> (f32)
  return %2 : f32
}
// CHECK-LABEL: func @simple_move()
//       CHECK:   %[[MOVED:.+]] = "moved_op"
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
func.func @move_region_dependencies() -> f32 {
  %0 = "before"() : () -> (f32)
  %1 = "moved_op"() : () -> (f32)
  %2 = "foo"() ({
    %3 = "inner_op"(%1) : (f32) -> (f32)
    "yield"(%3) : (f32) -> ()
  }) : () -> (f32)
  return %2 : f32
}
// CHECK-LABEL: func @move_region_dependencies()
//       CHECK:   %[[MOVED:.+]] = "moved_op"
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
func.func @move_in_topological_sort_order() -> f32 {
  %0 = "before"() : () -> (f32)
  %1 = "moved_op_1"() : () -> (f32)
  %2 = "moved_op_2"() : () -> (f32)
  %3 = "moved_op_3"(%1) : (f32) -> (f32)
  %4 = "moved_op_4"(%1, %3) : (f32, f32) -> (f32)
  %5 = "moved_op_5"(%2) : (f32) -> (f32)
  %6 = "foo"(%4, %5) : (f32, f32) -> (f32)
  return %6 : f32
}
// CHECK-LABEL: func @move_in_topological_sort_order()
//       CHECK:   %[[MOVED_1:.+]] = "moved_op_1"
//   CHECK-DAG:   %[[MOVED_2:.+]] = "moved_op_3"(%[[MOVED_1]])
//   CHECK-DAG:   %[[MOVED_3:.+]] = "moved_op_4"(%[[MOVED_1]], %[[MOVED_2]])
//   CHECK-DAG:   %[[MOVED_4:.+]] = "moved_op_2"
//   CHECK-DAG:   %[[MOVED_5:.+]] = "moved_op_5"(%[[MOVED_4]])
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

func.func @move_region_dependencies() -> f32 {
  %0 = "before"() : () -> (f32)
  %1 = "moved_op_1"() : () -> (f32)
  %2 = "moved_op_2"() ({
    "yield"(%1) : (f32) -> ()
  }) : () -> (f32)
  %3 = "foo"() ({
    "yield"(%2) : (f32) -> ()
  }) : () -> (f32)
  return %3 : f32
}

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
// CHECK-LABEL: func @move_region_dependencies()
//       CHECK:   %[[MOVED_1:.+]] = "moved_op_1"
//       CHECK:   %[[MOVED_2:.+]] = "moved_op_2"
//       CHECK:     "yield"(%[[MOVED_1]])
//       CHECK:   "before"
//       CHECK:   %[[FOO:.+]] = "foo"
//       CHECK:   return %[[FOO]]

// -----

// Current implementation omits following basic block argument when
// computing slices. Verify that this gives expected result.
func.func @ignore_basic_block_arguments() -> f32 {
^bb0():
  %8 = "test"() : () -> (f32)
  return %8: f32
^bb1(%bbArg : f32):
  %0 = "before"() : () -> (f32)
  %1 = "moved_op"() ({
    "yield"(%bbArg) : (f32) -> ()
  }) : () -> (f32)
  %2 = "foo"() ({
    "yield"(%1) : (f32) -> ()
  }) : () -> (f32)
  return %2 : f32
}

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
// CHECK-LABEL: func @ignore_basic_block_arguments()
//       CHECK:   %[[MOVED_1:.+]] = "moved_op"
//       CHECK:   "before"
//       CHECK:   %[[FOO:.+]] = "foo"
//       CHECK:   return %[[FOO]]

// -----

// Fail when the "before" operation is part of the operation slice.
func.func @do_not_move_slice() -> f32 {
  %0 = "before"() : () -> (f32)
  %1 = "moved_op"(%0) : (f32) -> (f32)
  %2 = "foo"(%1) : (f32) -> (f32)
  return %2 : f32
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0 : !transform.any_op {transform.readonly}) {
    %op1 = transform.structured.match ops{["foo"]} in %arg0
        : (!transform.any_op) -> !transform.any_op
    %op2 = transform.structured.match ops{["before"]} in %arg0
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
func.func @do_not_move_slice() -> f32 {
  %0 = "before"() : () -> (f32)
  %1 = "moved_op"() ({
    "yield"(%0) : (f32) -> ()
  }) : () -> (f32)
  %2 = "foo"() ({
    "yield"(%1) : (f32) -> ()
  }) : () -> (f32)
  return %2 : f32
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0 : !transform.any_op {transform.readonly}) {
    %op1 = transform.structured.match ops{["foo"]} in %arg0
        : (!transform.any_op) -> !transform.any_op
    %op2 = transform.structured.match ops{["before"]} in %arg0
        : (!transform.any_op) -> !transform.any_op
    // expected-remark@+1{{cannot move dependencies before operation in backward slice of op}}
    transform.test.move_operand_deps %op1 before %op2
        : !transform.any_op, !transform.any_op
    transform.yield
  }
}

// -----

// Dont move ops when insertion point does not dominate the op
func.func @do_not_move() -> f32 {
  %1 = "moved_op"() : () -> (f32)
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
func.func @simple_move_values() -> f32 {
  %0 = "before"() : () -> (f32)
  %1 = "moved_op_1"() : () -> (f32)
  %2 = "moved_op_2"() : () -> (f32)
  %3 = "foo"(%1, %2) : (f32, f32) -> (f32)
  return %3 : f32
}
// CHECK-LABEL: func @simple_move_values()
//       CHECK:   %[[MOVED1:.+]] = "moved_op_1"
//       CHECK:   %[[MOVED2:.+]] = "moved_op_2"
//       CHECK:   %[[BEFORE:.+]] = "before"
//       CHECK:   %[[FOO:.+]] = "foo"(%[[MOVED1]], %[[MOVED2]])
//       CHECK:   return %[[FOO]]

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0 : !transform.any_op {transform.readonly}) {
    %op1 = transform.structured.match ops{["moved_op_1"]} in %arg0
        : (!transform.any_op) -> !transform.any_op
    %op2 = transform.structured.match ops{["moved_op_2"]} in %arg0
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
func.func @move_region_dependencies_values() -> f32 {
  %0 = "before"() : () -> (f32)
  %1 = "moved_op_1"() : () -> (f32)
  %2 = "moved_op_2"() ({
    %3 = "inner_op"(%1) : (f32) -> (f32)
    "yield"(%3) : (f32) -> ()
  }) : () -> (f32)
  return %2 : f32
}
// CHECK-LABEL: func @move_region_dependencies_values()
//       CHECK:   %[[MOVED1:.+]] = "moved_op_1"
//       CHECK:   %[[MOVED2:.+]] = "moved_op_2"
//       CHECK:   %[[BEFORE:.+]] = "before"

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0 : !transform.any_op {transform.readonly}) {
    %op1 = transform.structured.match ops{["moved_op_2"]} in %arg0
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
func.func @move_values_in_topological_sort_order() -> f32 {
  %0 = "before"() : () -> (f32)
  %1 = "moved_op_1"() : () -> (f32)
  %2 = "moved_op_2"() : () -> (f32)
  %3 = "moved_op_3"(%1) : (f32) -> (f32)
  %4 = "moved_op_4"(%1, %3) : (f32, f32) -> (f32)
  %5 = "moved_op_5"(%2) : (f32) -> (f32)
  %6 = "foo"(%4, %5) : (f32, f32) -> (f32)
  return %6 : f32
}
// CHECK-LABEL: func @move_values_in_topological_sort_order()
//       CHECK:   %[[MOVED_1:.+]] = "moved_op_1"
//   CHECK-DAG:   %[[MOVED_2:.+]] = "moved_op_3"(%[[MOVED_1]])
//   CHECK-DAG:   %[[MOVED_3:.+]] = "moved_op_4"(%[[MOVED_1]], %[[MOVED_2]])
//   CHECK-DAG:   %[[MOVED_4:.+]] = "moved_op_2"
//   CHECK-DAG:   %[[MOVED_5:.+]] = "moved_op_5"(%[[MOVED_4]])
//       CHECK:   %[[BEFORE:.+]] = "before"
//       CHECK:   %[[FOO:.+]] = "foo"(%[[MOVED_3]], %[[MOVED_5]])
//       CHECK:   return %[[FOO]]

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0 : !transform.any_op {transform.readonly}) {
    %op1 = transform.structured.match ops{["moved_op_4"]} in %arg0
        : (!transform.any_op) -> !transform.any_op
    %op2 = transform.structured.match ops{["moved_op_5"]} in %arg0
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

func.func @move_only_required_defns() -> (f32, f32, f32, f32) {
  %0 = "unmoved_op"() : () -> (f32)
  %1 = "dummy_op"() : () -> (f32)
  %2 = "before"() : () -> (f32)
  %3 = "moved_op"() : () -> (f32)
  return %0, %1, %2, %3 : f32, f32, f32, f32
}
// CHECK-LABEL: func @move_only_required_defns()
//       CHECK:   %[[UNMOVED:.+]] = "unmoved_op"
//       CHECK:   %[[DUMMY:.+]] = "dummy_op"
//       CHECK:   %[[MOVED:.+]] = "moved_op"
//       CHECK:   %[[BEFORE:.+]] = "before"

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0 : !transform.any_op {transform.readonly}) {
    %op1 = transform.structured.match ops{["unmoved_op"]} in %arg0
        : (!transform.any_op) -> !transform.any_op
    %op2 = transform.structured.match ops{["dummy_op"]} in %arg0
        : (!transform.any_op) -> !transform.any_op
    %op3 = transform.structured.match ops{["before"]} in %arg0
        : (!transform.any_op) -> !transform.any_op
    %op4 = transform.structured.match ops{["moved_op"]} in %arg0
        : (!transform.any_op) -> !transform.any_op
    %v1 = transform.get_result %op1[0] : (!transform.any_op) -> !transform.any_value
    %v2 = transform.get_result %op4[0] : (!transform.any_op) -> !transform.any_value
    transform.test.move_value_defns %v1, %v2 before %op3
        : (!transform.any_value, !transform.any_value), !transform.any_op
    transform.yield
  }
}

// -----

// Move only those value definitions that are not dominated by insertion point

func.func @move_only_required_defns() -> (f32, f32, f32, f32) {
  %0 = "unmoved_op"() : () -> (f32)
  %1 = "dummy_op"() : () -> (f32)
  %2 = "before"() : () -> (f32)
  %3 = "moved_op"() : () -> (f32)
  return %0, %1, %2, %3 : f32, f32, f32, f32
}
// CHECK-LABEL: func @move_only_required_defns()
//       CHECK:   %[[UNMOVED:.+]] = "unmoved_op"
//       CHECK:   %[[DUMMY:.+]] = "dummy_op"
//       CHECK:   %[[MOVED:.+]] = "moved_op"
//       CHECK:   %[[BEFORE:.+]] = "before"

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0 : !transform.any_op {transform.readonly}) {
    %op1 = transform.structured.match ops{["unmoved_op"]} in %arg0
        : (!transform.any_op) -> !transform.any_op
    %op2 = transform.structured.match ops{["dummy_op"]} in %arg0
        : (!transform.any_op) -> !transform.any_op
    %op3 = transform.structured.match ops{["before"]} in %arg0
        : (!transform.any_op) -> !transform.any_op
    %op4 = transform.structured.match ops{["moved_op"]} in %arg0
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
func.func @move_only_required_defns() -> (f32, f32) {
  %0 = "unmoved_op"() : () -> (f32)
  cf.br ^bb0(%0 : f32) 
 ^bb0(%arg0 : f32) :
  %1 = "before"() : () -> (f32)
  %2 = "moved_op"(%arg0) : (f32) -> (f32)
  return %1, %2 : f32, f32
}
// CHECK-LABEL: func @move_only_required_defns()
//       CHECK:   %[[MOVED:.+]] = "moved_op"
//       CHECK:   %[[BEFORE:.+]] = "before"

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0 : !transform.any_op {transform.readonly}) {
    %op1 = transform.structured.match ops{["before"]} in %arg0
        : (!transform.any_op) -> !transform.any_op
    %op2 = transform.structured.match ops{["moved_op"]} in %arg0
        : (!transform.any_op) -> !transform.any_op
    %v1 = transform.get_result %op2[0] : (!transform.any_op) -> !transform.any_value
    transform.test.move_value_defns %v1 before %op1
        : (!transform.any_value), !transform.any_op
    transform.yield
  }
}

// -----

// Do not move across basic blocks
func.func @no_move_across_basic_blocks() -> (f32, f32) {
  %0 = "unmoved_op"() : () -> (f32)
  %1 = "before"() : () -> (f32)
  cf.br ^bb0(%0 : f32) 
 ^bb0(%arg0 : f32) :
  %2 = "moved_op"(%arg0) : (f32) -> (f32)
  return %1, %2 : f32, f32
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0 : !transform.any_op {transform.readonly}) {
    %op1 = transform.structured.match ops{["before"]} in %arg0
        : (!transform.any_op) -> !transform.any_op
    %op2 = transform.structured.match ops{["moved_op"]} in %arg0
        : (!transform.any_op) -> !transform.any_op
    %v1 = transform.get_result %op2[0] : (!transform.any_op) -> !transform.any_value
    // expected-remark@+1{{unsupported case of moving definition of value before an insertion point in a different basic block}}
    transform.test.move_value_defns %v1 before %op1
        : (!transform.any_value), !transform.any_op
    transform.yield
  }
}
