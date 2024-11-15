// RUN: mlir-opt %s --transform-interpreter --split-input-file | FileCheck %s

///----------------------------------------------------------------------------------------
/// [Pattern: ShapeCastOpFolder]
///----------------------------------------------------------------------------------------

// CHECK-LABEL: func @fixed_width
//  CHECK-SAME: %[[A0:.*0]]: vector<2x4xf32>
//   CHECK-NOT: vector.shape_cast
//       CHECK: return %[[A0]] : vector<2x4xf32>
func.func @fixed_width(%arg0 : vector<2x4xf32>) -> vector<2x4xf32> {
  %0 = vector.shape_cast %arg0 : vector<2x4xf32> to vector<8xf32>
  %1 = vector.shape_cast %0 : vector<8xf32> to vector<2x4xf32>
  return %1 : vector<2x4xf32>
}

// CHECK-LABEL: func @scalable
//  CHECK-SAME: %[[A0:.*0]]: vector<2x[4]xf32>
//   CHECK-NOT: vector.shape_cast
//       CHECK: return %[[A0]] : vector<2x[4]xf32>
func.func @scalable(%arg0 : vector<2x[4]xf32>) -> vector<2x[4]xf32> {
  %0 = vector.shape_cast %arg0 : vector<2x[4]xf32> to vector<[8]xf32>
  %1 = vector.shape_cast %0 : vector<[8]xf32> to vector<2x[4]xf32>
  return %1 : vector<2x[4]xf32>
}

// ============================================================================
//  TD sequence
// ============================================================================
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root : !transform.any_op {transform.readonly}) {
    %func_op = transform.structured.match ops{["func.func"]} in %root : (!transform.any_op) -> !transform.op<"func.func">
    transform.apply_patterns to %func_op {
      transform.apply_patterns.vector.drop_unit_dims_with_shape_cast
    } : !transform.op<"func.func">
    transform.yield
  }
}
