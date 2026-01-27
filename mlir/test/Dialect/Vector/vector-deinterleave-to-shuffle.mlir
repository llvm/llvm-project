// RUN: mlir-opt %s --transform-interpreter | FileCheck %s

// CHECK-LABEL: @vector_deinterleave_to_shuffle
func.func @vector_deinterleave_to_shuffle(%arg0: vector<14xi16>) -> (vector<7xi16>, vector<7xi16>) {
  %evens, %odds = vector.deinterleave %arg0 : vector<14xi16> -> vector<7xi16>
  return %evens, %odds : vector<7xi16>, vector<7xi16>
}
// CHECK: vector.shuffle %arg0, %arg0 [0, 2, 4, 6, 8, 10, 12] : vector<14xi16>, vector<14xi16>
// CHECK: vector.shuffle %arg0, %arg0 [1, 3, 5, 7, 9, 11, 13] : vector<14xi16>, vector<14xi16>

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%module_op: !transform.any_op {transform.readonly}) {
    %f = transform.structured.match ops{["func.func"]} in %module_op
      : (!transform.any_op) -> !transform.any_op

    transform.apply_patterns to %f {
      transform.apply_patterns.vector.deinterleave_to_shuffle
    } : !transform.any_op
    transform.yield
  }
}
