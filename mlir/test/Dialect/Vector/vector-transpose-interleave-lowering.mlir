// RUN: mlir-opt %s --transform-interpreter --split-input-file | FileCheck %s

// CHECK-LABEL: func.func @rows_to_columns(
// CHECK-NOT: vector.transpose
// CHECK-NOT: memref.alloca
// CHECK-NOT: vector.vscale
// CHECK: %[[FLAT:.*]] = vector.shape_cast %{{.*}} : vector<8x2xf32> to vector<16xf32>
// CHECK: %[[EVEN:.*]], %[[ODD:.*]] = vector.deinterleave %[[FLAT]] : vector<16xf32> -> vector<8xf32>
// CHECK: %[[JOINED:.*]] = vector.shuffle %[[EVEN]], %[[ODD]]
// CHECK: %[[RESULT:.*]] = vector.shape_cast %[[JOINED]] : vector<16xf32> to vector<2x8xf32>
// CHECK: return %[[RESULT]]
func.func @rows_to_columns(%arg: vector<8x2xf32>) -> vector<2x8xf32> {
  %result = vector.transpose %arg, [1, 0] : vector<8x2xf32> to vector<2x8xf32>
  return %result : vector<2x8xf32>
}

// CHECK-LABEL: func.func @columns_to_rows(
// CHECK-NOT: vector.transpose
// CHECK-NOT: memref.alloca
// CHECK-NOT: vector.vscale
// CHECK: %[[LHS:.*]] = vector.extract %{{.*}}[0] : vector<8xf32> from vector<2x8xf32>
// CHECK: %[[RHS:.*]] = vector.extract %{{.*}}[1] : vector<8xf32> from vector<2x8xf32>
// CHECK: %[[PAIR:.*]] = vector.interleave %[[LHS]], %[[RHS]] : vector<8xf32> -> vector<16xf32>
// CHECK: %[[RESULT:.*]] = vector.shape_cast %[[PAIR]] : vector<16xf32> to vector<8x2xf32>
// CHECK: return %[[RESULT]]
func.func @columns_to_rows(%arg: vector<2x8xf32>) -> vector<8x2xf32> {
  %result = vector.transpose %arg, [1, 0] : vector<2x8xf32> to vector<8x2xf32>
  return %result : vector<8x2xf32>
}

// CHECK-LABEL: func.func @integer_elements(
// CHECK: vector.deinterleave
func.func @integer_elements(%arg: vector<8x2xi16>) -> vector<2x8xi16> {
  %result = vector.transpose %arg, [1, 0] : vector<8x2xi16> to vector<2x8xi16>
  return %result : vector<2x8xi16>
}

// CHECK-LABEL: func.func @no_dimension_of_size_two(
// CHECK: vector.transpose
func.func @no_dimension_of_size_two(%arg: vector<8x3xf32>) -> vector<3x8xf32> {
  %result = vector.transpose %arg, [1, 0] : vector<8x3xf32> to vector<3x8xf32>
  return %result : vector<3x8xf32>
}

// Rank greater than two is deliberately left for a future extension.
// CHECK-LABEL: func.func @batched(
// CHECK: vector.transpose
func.func @batched(%arg: vector<4x8x2xf32>) -> vector<4x2x8xf32> {
  %result = vector.transpose %arg, [0, 2, 1]
      : vector<4x8x2xf32> to vector<4x2x8xf32>
  return %result : vector<4x2x8xf32>
}

// CHECK-LABEL: func.func @already_scalable(
// CHECK: vector.transpose
func.func @already_scalable(%arg: vector<2x[4]xf32>) -> vector<[4]x2xf32> {
  %result = vector.transpose %arg, [1, 0] : vector<2x[4]xf32> to vector<[4]x2xf32>
  return %result : vector<[4]x2xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
      %root : !transform.any_op {transform.readonly}) {
    %funcs = transform.structured.match ops{["func.func"]} in %root
      : (!transform.any_op) -> !transform.op<"func.func">
    transform.apply_patterns to %funcs {
      transform.apply_patterns.vector.lower_transpose_to_interleave
    } : !transform.op<"func.func">
    transform.yield
  }
}
