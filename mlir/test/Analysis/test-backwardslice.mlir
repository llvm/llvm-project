// RUN: mlir-opt --allow-unregistered-dialect --transform-interpreter --split-input-file --verify-diagnostics %s | FileCheck %s

func.func @simple() {
  %0 = "other"() : () -> (f32)
  %1 = "root"(%0) : (f32) -> (f32)
}
// CHECK-LABEL: func @simple__backward_slice__()
//       CHECK:   %[[OTHER:.+]] = "other"
//       CHECK:   %[[ROOT:.+]] = "root"(%[[OTHER]])
//       CHECK:   return

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0 : !transform.any_op {transform.readonly}) {
    %op = transform.structured.match ops{["root"]} in %arg0
        : (!transform.any_op) -> !transform.any_op
    transform.test.get_backward_slice %op : !transform.any_op
    transform.yield
  }
}

// -----

func.func @across_blocks() {
  %0 = "other"() : () -> (f32)
  cf.br ^bb1
^bb1() :
  %1 = "root"(%0) : (f32) -> (f32)
}
// CHECK-LABEL: func @across_blocks__backward_slice__()
//       CHECK:   %[[OTHER:.+]] = "other"
//       CHECK:   %[[ROOT:.+]] = "root"(%[[OTHER]])
//       CHECK:   return

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0 : !transform.any_op {transform.readonly}) {
    %op = transform.structured.match ops{["root"]} in %arg0
        : (!transform.any_op) -> !transform.any_op
    transform.test.get_backward_slice %op : !transform.any_op
    transform.yield
  }
}

// -----

func.func @large_slice() {
  %0 = "not_in_slice"() : () -> (f32)
  %1 = "sliced_op0"() : () -> (f32)
  %2 = "sliced_op1"() : () -> (f32)
  %3 = "sliced_op"(%1, %2) : (f32, f32) -> (f32)
  %4 = "not_in_slice"() : () -> (f32)
  %5 = "root"(%3) : (f32) -> (f32)
  %6 = "not_in_slice"() : () -> (f32)
}
// CHECK-LABEL: func @large_slice__backward_slice__()
//   CHECK-NOT:   "not_in_slice"
//   CHECK-DAG:   %[[OP0:.+]] = "sliced_op0"
//   CHECK-DAG:   %[[OP1:.+]] = "sliced_op1"
//   CHECK-NOT:   "not_in_slice"
//       CHECK:   %[[OP2:.+]] = "sliced_op"(%[[OP0]], %[[OP1]])
//       CHECK:   %[[ROOT:.+]] = "root"(%[[OP2]])
//   CHECK-NOT:   "not_in_slice"
//       CHECK:   return

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0 : !transform.any_op {transform.readonly}) {
    %op = transform.structured.match ops{["root"]} in %arg0
        : (!transform.any_op) -> !transform.any_op
    transform.test.get_backward_slice %op : !transform.any_op
    transform.yield
  }
}

// -----

func.func @include_uses_from_above() {
  %0 = "sliced_op"() : () -> (f32)
  %1 = "sliced_op" () ({
  ^bb0():
    "yield" (%0) : (f32) -> ()
  }): () -> (f32)
  %2 = "root"(%1) : (f32) -> (f32)
}
// CHECK-LABEL: func @include_uses_from_above__backward_slice__()
//       CHECK:   %[[OP0:.+]] = "sliced_op"
//       CHECK:   %[[OP1:.+]] = "sliced_op"
//  CHECK-NEXT:     "yield"(%[[OP0]])
//       CHECK:   %[[ROOT:.+]] = "root"(%[[OP1]])
//       CHECK:   return

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0 : !transform.any_op {transform.readonly}) {
    %op = transform.structured.match ops{["root"]} in %arg0
        : (!transform.any_op) -> !transform.any_op
    transform.test.get_backward_slice %op : !transform.any_op
    transform.yield
  }
}
