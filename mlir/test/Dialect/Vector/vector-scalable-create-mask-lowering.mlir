// RUN: mlir-opt %s --test-transform-dialect-interpreter | FileCheck %s

// CHECK-LABEL: func.func @create_mask_2d_trailing_scalable(
// CHECK-SAME: %[[arg:.*]]: index) -> vector<3x[4]xi1> {
// CHECK-NEXT: %[[zero_mask_1d:.*]] = arith.constant dense<false> : vector<[4]xi1>
// CHECK-NEXT: %[[zero_mask_2d:.*]] = arith.constant dense<false> : vector<3x[4]xi1>
// CHECK-NEXT: %[[create_mask_1d:.*]] = vector.create_mask %[[arg]] : vector<[4]xi1>
// CHECK-NEXT: %[[res_0:.*]] = vector.insert %[[create_mask_1d]], %[[zero_mask_2d]] [0] : vector<[4]xi1> into vector<3x[4]xi1>
// CHECK-NEXT: %[[res_1:.*]] = vector.insert %[[create_mask_1d]], %[[res_0]] [1] : vector<[4]xi1> into vector<3x[4]xi1>
// CHECK-NEXT: %[[res_2:.*]] = vector.insert %[[zero_mask_1d]], %[[res_1]] [2] : vector<[4]xi1> into vector<3x[4]xi1>
// CHECK-NEXT: return %[[res_2]] : vector<3x[4]xi1>
func.func @create_mask_2d_trailing_scalable(%a: index) -> vector<3x[4]xi1> {
  %c2 = arith.constant 2 : index
  %mask = vector.create_mask %c2, %a : vector<3x[4]xi1>
  return %mask : vector<3x[4]xi1>
}

// -----

/// The following cannot be lowered as the current lowering requires unrolling
/// the leading dim.

// CHECK-LABEL: func.func @cannot_create_mask_2d_leading_scalable(
// CHECK-SAME: %[[arg:.*]]: index) -> vector<[4]x4xi1> {
// CHECK: %{{.*}} = vector.create_mask %[[arg]], %{{.*}} : vector<[4]x4xi1>
func.func @cannot_create_mask_2d_leading_scalable(%a: index) -> vector<[4]x4xi1> {
  %c1 = arith.constant 1 : index
  %mask = vector.create_mask %a, %c1 : vector<[4]x4xi1>
  return %mask : vector<[4]x4xi1>
}

transform.sequence failures(suppress) {
^bb1(%module_op: !transform.any_op):
  %f = transform.structured.match ops{["func.func"]} in %module_op
    : (!transform.any_op) -> !transform.any_op

  transform.apply_patterns to %f {
    transform.apply_patterns.vector.lower_create_mask
  } : !transform.any_op
}
