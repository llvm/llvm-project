// RUN: mlir-opt %s --transform-interpreter | FileCheck %s

// CHECK-LABEL: @vector_interleave_to_shuffle_1d
func.func @vector_interleave_to_shuffle_1d(%a: vector<7xi16>, %b: vector<7xi16>) -> vector<14xi16> {
  %0 = vector.interleave %a, %b : vector<7xi16> -> vector<14xi16>
  return %0 : vector<14xi16>
}
// CHECK: vector.shuffle %arg0, %arg1 [0, 7, 1, 8, 2, 9, 3, 10, 4, 11, 5, 12, 6, 13] : vector<7xi16>, vector<7xi16>

// CHECK-LABEL: @vector_interleave_to_shuffle_0d
func.func @vector_interleave_to_shuffle_0d(%a: vector<f32>, %b: vector<f32>) -> vector<2xf32> {
  %0 = vector.interleave %a, %b : vector<f32> -> vector<2xf32>
  return %0 : vector<2xf32>
}
// CHECK: vector.shuffle %arg0, %arg1 [0, 1] : vector<f32>, vector<f32>

// CHECK-LABEL: @vector_deinterleave_to_shuffle_1d
func.func @vector_deinterleave_to_shuffle_1d(%arg0: vector<14xi16>) -> (vector<7xi16>, vector<7xi16>) {
  %evens, %odds = vector.deinterleave %arg0 : vector<14xi16> -> vector<7xi16>
  return %evens, %odds : vector<7xi16>, vector<7xi16>
}
// CHECK: vector.shuffle %arg0, %arg0 [0, 2, 4, 6, 8, 10, 12] : vector<14xi16>, vector<14xi16>
// CHECK: vector.shuffle %arg0, %arg0 [1, 3, 5, 7, 9, 11, 13] : vector<14xi16>, vector<14xi16>

// CHECK-LABEL: @vector_deinterleave_size2
func.func @vector_deinterleave_size2(%arg0: vector<2xi32>) -> (vector<1xi32>, vector<1xi32>) {
  %evens, %odds = vector.deinterleave %arg0 : vector<2xi32> -> vector<1xi32>
  return %evens, %odds : vector<1xi32>, vector<1xi32>
}
// CHECK: vector.shuffle %arg0, %arg0 [0] : vector<2xi32>, vector<2xi32>
// CHECK: vector.shuffle %arg0, %arg0 [1] : vector<2xi32>, vector<2xi32>

// CHECK-LABEL: @negative_cases
// CHECK-NOT: vector.shuffle
func.func @negative_cases(
    %a: vector<[4]xi32>, %b: vector<[4]xi32>,
    %c: vector<2x4xi32>, %d: vector<2x4xi32>,
    %e: vector<[8]xi32>,
    %f: vector<2x8xi32>) -> (vector<[8]xi32>, vector<2x8xi32>,
                              vector<[4]xi32>, vector<[4]xi32>,
                              vector<2x4xi32>, vector<2x4xi32>) {
  %0 = vector.interleave %a, %b : vector<[4]xi32> -> vector<[8]xi32>
  %1 = vector.interleave %c, %d : vector<2x4xi32> -> vector<2x8xi32>
  %evens0, %odds0 = vector.deinterleave %e : vector<[8]xi32> -> vector<[4]xi32>
  %evens1, %odds1 = vector.deinterleave %f : vector<2x8xi32> -> vector<2x4xi32>
  return %0, %1, %evens0, %odds0, %evens1, %odds1
    : vector<[8]xi32>, vector<2x8xi32>, vector<[4]xi32>, vector<[4]xi32>, vector<2x4xi32>, vector<2x4xi32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%module_op: !transform.any_op {transform.readonly}) {
    %f = transform.structured.match ops{["func.func"]} in %module_op
      : (!transform.any_op) -> !transform.any_op

    transform.apply_patterns to %f {
      transform.apply_patterns.vector.interleave_and_deinterleave_to_shuffle
    } : !transform.any_op
    transform.yield
  }
}
