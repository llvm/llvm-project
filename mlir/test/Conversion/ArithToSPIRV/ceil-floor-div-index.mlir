// RUN: mlir-opt %s -arith-expand -convert-arith-to-spirv | FileCheck %s

// CHECK-LABEL: func.func @ceildivui_index
// CHECK: %[[Q:.*]] = spirv.UDiv
// CHECK: %[[PRODUCT:.*]] = spirv.IMul %[[Q]]
// CHECK: %[[INEXACT:.*]] = spirv.INotEqual %[[PRODUCT]]
// CHECK: %[[ADJUSTMENT:.*]] = spirv.Select %[[INEXACT]]
// CHECK: %[[RESULT:.*]] = spirv.IAdd %[[Q]], %[[ADJUSTMENT]]
// CHECK: return %{{.*}} : index
func.func @ceildivui_index(%lhs: index, %rhs: index) -> index {
  %result = arith.ceildivui %lhs, %rhs : index
  return %result : index
}

// CHECK-LABEL: func.func @ceildivsi_index
// CHECK: %[[Q:.*]] = spirv.SDiv
// CHECK: %[[PRODUCT:.*]] = spirv.IMul %[[Q]]
// CHECK: %[[INEXACT:.*]] = spirv.INotEqual %[[PRODUCT]]
// CHECK: %[[SIGNED_LT:.*]] = spirv.SLessThan
// CHECK: %[[UNSIGNED_LT:.*]] = spirv.ULessThan
// CHECK: %[[SAME_SIGN:.*]] = spirv.LogicalEqual %[[SIGNED_LT]], %[[UNSIGNED_LT]]
// CHECK: %[[ROUND:.*]] = spirv.LogicalAnd %[[INEXACT]], %[[SAME_SIGN]]
// CHECK: %[[ADJUSTMENT:.*]] = spirv.Select %[[ROUND]]
// CHECK: %[[RESULT:.*]] = spirv.IAdd %[[Q]], %[[ADJUSTMENT]]
// CHECK: return %{{.*}} : index
func.func @ceildivsi_index(%lhs: index, %rhs: index) -> index {
  %result = arith.ceildivsi %lhs, %rhs : index
  return %result : index
}

// CHECK-LABEL: func.func @floordivsi_index
// CHECK: %[[Q:.*]] = spirv.SDiv
// CHECK: %[[PRODUCT:.*]] = spirv.IMul %[[Q]]
// CHECK: %[[INEXACT:.*]] = spirv.INotEqual %[[PRODUCT]]
// CHECK: %[[SIGNED_LT:.*]] = spirv.SLessThan
// CHECK: %[[UNSIGNED_LT:.*]] = spirv.ULessThan
// CHECK: %[[OPPOSITE_SIGN:.*]] = spirv.LogicalNotEqual %[[SIGNED_LT]], %[[UNSIGNED_LT]]
// CHECK: %[[ROUND:.*]] = spirv.LogicalAnd %[[INEXACT]], %[[OPPOSITE_SIGN]]
// CHECK: %[[ADJUSTMENT:.*]] = spirv.Select %[[ROUND]]
// CHECK: %[[RESULT:.*]] = spirv.ISub %[[Q]], %[[ADJUSTMENT]]
// CHECK: return %{{.*}} : index
func.func @floordivsi_index(%lhs: index, %rhs: index) -> index {
  %result = arith.floordivsi %lhs, %rhs : index
  return %result : index
}
