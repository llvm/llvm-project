// RUN: mlir-opt %s --transform-interpreter | FileCheck %s

// CHECK-LABEL: @vector_interleave_to_shuffle
func.func @vector_interleave_to_shuffle(%a: vector<7xi16>, %b: vector<7xi16>) -> vector<14xi16> {
  %0 = vector.interleave %a, %b : vector<7xi16> -> vector<14xi16>
  return %0 : vector<14xi16>
}
// CHECK: vector.shuffle %arg0, %arg1 [0, 7, 1, 8, 2, 9, 3, 10, 4, 11, 5, 12, 6, 13] : vector<7xi16>, vector<7xi16>

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%module_op: !transform.any_op {transform.readonly}) {
    %f = transform.structured.match ops{["func.func"]} in %module_op
      : (!transform.any_op) -> !transform.any_op

    transform.apply_patterns to %f {
      transform.apply_patterns.vector.interleave_to_shuffle
    } : !transform.any_op
    transform.yield
  }
}
