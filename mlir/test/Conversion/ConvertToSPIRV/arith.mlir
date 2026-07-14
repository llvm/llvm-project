// RUN: mlir-opt -test-convert-to-spirv="run-signature-conversion=false run-vector-unrolling=false" -split-input-file %s | FileCheck %s

//===----------------------------------------------------------------------===//
// arithmetic ops
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @int32_scalar
func.func @int32_scalar(%lhs: i32, %rhs: i32) {
  // CHECK: spirv.IAdd %{{.*}}, %{{.*}}: i32
  %0 = arith.addi %lhs, %rhs: i32
  // CHECK: spirv.ISub %{{.*}}, %{{.*}}: i32
  %1 = arith.subi %lhs, %rhs: i32
  // CHECK: spirv.IMul %{{.*}}, %{{.*}}: i32
  %2 = arith.muli %lhs, %rhs: i32
  // CHECK: spirv.SDiv %{{.*}}, %{{.*}}: i32
  %3 = arith.divsi %lhs, %rhs: i32
  // CHECK: spirv.UDiv %{{.*}}, %{{.*}}: i32
  %4 = arith.divui %lhs, %rhs: i32
  // CHECK: spirv.UMod %{{.*}}, %{{.*}}: i32
  %5 = arith.remui %lhs, %rhs: i32
  return
}

// CHECK-LABEL: @int32_scalar_srem
// CHECK-SAME: (%[[LHS:.+]]: i32, %[[RHS:.+]]: i32)
func.func @int32_scalar_srem(%lhs: i32, %rhs: i32) {
  // CHECK: %[[LABS:.+]] = spirv.GL.SAbs %[[LHS]] : i32
  // CHECK: %[[RABS:.+]] = spirv.GL.SAbs %[[RHS]] : i32
  // CHECK:  %[[ABS:.+]] = spirv.UMod %[[LABS]], %[[RABS]] : i32
  // CHECK:  %[[POS:.+]] = spirv.IEqual %[[LHS]], %[[LABS]] : i32
  // CHECK:  %[[NEG:.+]] = spirv.SNegate %[[ABS]] : i32
  // CHECK:      %{{.+}} = spirv.Select %[[POS]], %[[ABS]], %[[NEG]] : i1, i32
  %0 = arith.remsi %lhs, %rhs: i32
  return
}

// CHECK-LABEL: @scalar_ceildivui
// CHECK-SAME: (%[[LHS:.+]]: i32, %[[RHS:.+]]: i32)
func.func @scalar_ceildivui(%lhs: i32, %rhs: i32) -> i32 {
  // CHECK:     %[[Q:.+]] = spirv.UDiv %[[LHS]], %[[RHS]] : i32
  // CHECK:     %[[PRODUCT:.+]] = spirv.IMul %[[Q]], %[[RHS]] : i32
  // CHECK:     %[[INEXACT:.+]] = spirv.INotEqual %[[PRODUCT]], %[[LHS]] : i32
  // CHECK:     %[[ZERO:.+]] = spirv.Constant 0 : i32
  // CHECK:     %[[ONE:.+]] = spirv.Constant 1 : i32
  // CHECK:     %[[ADJUSTMENT:.+]] = spirv.Select %[[INEXACT]], %[[ONE]], %[[ZERO]] : i1, i32
  // CHECK:     %[[R:.+]] = spirv.IAdd %[[Q]], %[[ADJUSTMENT]] : i32
  // CHECK:     spirv.ReturnValue %[[R]]
  %0 = arith.ceildivui %lhs, %rhs : i32
  return %0 : i32
}

// CHECK-LABEL: @scalar_ceildivsi
// CHECK-SAME: (%[[LHS:.+]]: i32, %[[RHS:.+]]: i32)
func.func @scalar_ceildivsi(%lhs: i32, %rhs: i32) -> i32 {
  // CHECK:     %[[Q:.+]] = spirv.SDiv %[[LHS]], %[[RHS]] : i32
  // CHECK:     %[[PRODUCT:.+]] = spirv.IMul %[[Q]], %[[RHS]] : i32
  // CHECK:     %[[INEXACT:.+]] = spirv.INotEqual %[[PRODUCT]], %[[LHS]] : i32
  // CHECK:     %[[SIGNED_LT:.+]] = spirv.SLessThan %[[LHS]], %[[RHS]] : i32
  // CHECK:     %[[UNSIGNED_LT:.+]] = spirv.ULessThan %[[LHS]], %[[RHS]] : i32
  // CHECK:     %[[SAME_SIGN:.+]] = spirv.LogicalEqual %[[SIGNED_LT]], %[[UNSIGNED_LT]] : i1
  // CHECK:     %[[ROUND:.+]] = spirv.LogicalAnd %[[INEXACT]], %[[SAME_SIGN]] : i1
  // CHECK:     %[[ZERO:.+]] = spirv.Constant 0 : i32
  // CHECK:     %[[ONE:.+]] = spirv.Constant 1 : i32
  // CHECK:     %[[ADJUSTMENT:.+]] = spirv.Select %[[ROUND]], %[[ONE]], %[[ZERO]] : i1, i32
  // CHECK:     %[[R:.+]] = spirv.IAdd %[[Q]], %[[ADJUSTMENT]] : i32
  // CHECK:     spirv.ReturnValue %[[R]]
  %0 = arith.ceildivsi %lhs, %rhs : i32
  return %0 : i32
}

// CHECK-LABEL: @scalar_floordivsi
// CHECK-SAME: (%[[LHS:.+]]: i32, %[[RHS:.+]]: i32)
func.func @scalar_floordivsi(%lhs: i32, %rhs: i32) -> i32 {
  // CHECK:     %[[Q:.+]] = spirv.SDiv %[[LHS]], %[[RHS]] : i32
  // CHECK:     %[[PRODUCT:.+]] = spirv.IMul %[[Q]], %[[RHS]] : i32
  // CHECK:     %[[INEXACT:.+]] = spirv.INotEqual %[[PRODUCT]], %[[LHS]] : i32
  // CHECK:     %[[SIGNED_LT:.+]] = spirv.SLessThan %[[LHS]], %[[RHS]] : i32
  // CHECK:     %[[UNSIGNED_LT:.+]] = spirv.ULessThan %[[LHS]], %[[RHS]] : i32
  // CHECK:     %[[OPPOSITE_SIGN:.+]] = spirv.LogicalNotEqual %[[SIGNED_LT]], %[[UNSIGNED_LT]] : i1
  // CHECK:     %[[ROUND:.+]] = spirv.LogicalAnd %[[INEXACT]], %[[OPPOSITE_SIGN]] : i1
  // CHECK:     %[[ZERO:.+]] = spirv.Constant 0 : i32
  // CHECK:     %[[ONE:.+]] = spirv.Constant 1 : i32
  // CHECK:     %[[ADJUSTMENT:.+]] = spirv.Select %[[ROUND]], %[[ONE]], %[[ZERO]] : i1, i32
  // CHECK:     %[[R:.+]] = spirv.ISub %[[Q]], %[[ADJUSTMENT]] : i32
  // CHECK:     spirv.ReturnValue %[[R]]
  %0 = arith.floordivsi %lhs, %rhs : i32
  return %0 : i32
}

// CHECK-LABEL: @vector_ceildivsi
func.func @vector_ceildivsi(%lhs: vector<4xi32>, %rhs: vector<4xi32>) -> vector<4xi32> {
  // CHECK: spirv.SDiv %{{.*}}, %{{.*}} : vector<4xi32>
  // CHECK: %[[ADJUSTMENT:.+]] = spirv.Select %{{.*}}, %{{.*}}, %{{.*}} : vector<4xi1>, vector<4xi32>
  // CHECK: spirv.IAdd %{{.*}}, %[[ADJUSTMENT]] : vector<4xi32>
  %0 = arith.ceildivsi %lhs, %rhs : vector<4xi32>
  return %0 : vector<4xi32>
}

// -----

//===----------------------------------------------------------------------===//
// arith bit ops
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @bitwise_scalar
func.func @bitwise_scalar(%arg0 : i32, %arg1 : i32) {
  // CHECK: spirv.BitwiseAnd
  %0 = arith.andi %arg0, %arg1 : i32
  // CHECK: spirv.BitwiseOr
  %1 = arith.ori %arg0, %arg1 : i32
  // CHECK: spirv.BitwiseXor
  %2 = arith.xori %arg0, %arg1 : i32
  return
}

// CHECK-LABEL: @bitwise_vector
func.func @bitwise_vector(%arg0 : vector<4xi32>, %arg1 : vector<4xi32>) {
  // CHECK: spirv.BitwiseAnd
  %0 = arith.andi %arg0, %arg1 : vector<4xi32>
  // CHECK: spirv.BitwiseOr
  %1 = arith.ori %arg0, %arg1 : vector<4xi32>
  // CHECK: spirv.BitwiseXor
  %2 = arith.xori %arg0, %arg1 : vector<4xi32>
  return
}

// CHECK-LABEL: @logical_scalar
func.func @logical_scalar(%arg0 : i1, %arg1 : i1) {
  // CHECK: spirv.LogicalAnd
  %0 = arith.andi %arg0, %arg1 : i1
  // CHECK: spirv.LogicalOr
  %1 = arith.ori %arg0, %arg1 : i1
  // CHECK: spirv.LogicalNotEqual
  %2 = arith.xori %arg0, %arg1 : i1
  return
}

// CHECK-LABEL: @logical_vector
func.func @logical_vector(%arg0 : vector<4xi1>, %arg1 : vector<4xi1>) {
  // CHECK: spirv.LogicalAnd
  %0 = arith.andi %arg0, %arg1 : vector<4xi1>
  // CHECK: spirv.LogicalOr
  %1 = arith.ori %arg0, %arg1 : vector<4xi1>
  // CHECK: spirv.LogicalNotEqual
  %2 = arith.xori %arg0, %arg1 : vector<4xi1>
  return
}

// CHECK-LABEL: @shift_scalar
func.func @shift_scalar(%arg0 : i32, %arg1 : i32) {
  // CHECK: spirv.ShiftLeftLogical
  %0 = arith.shli %arg0, %arg1 : i32
  // CHECK: spirv.ShiftRightArithmetic
  %1 = arith.shrsi %arg0, %arg1 : i32
  // CHECK: spirv.ShiftRightLogical
  %2 = arith.shrui %arg0, %arg1 : i32
  return
}

// CHECK-LABEL: @shift_vector
func.func @shift_vector(%arg0 : vector<4xi32>, %arg1 : vector<4xi32>) {
  // CHECK: spirv.ShiftLeftLogical
  %0 = arith.shli %arg0, %arg1 : vector<4xi32>
  // CHECK: spirv.ShiftRightArithmetic
  %1 = arith.shrsi %arg0, %arg1 : vector<4xi32>
  // CHECK: spirv.ShiftRightLogical
  %2 = arith.shrui %arg0, %arg1 : vector<4xi32>
  return
}

// -----

//===----------------------------------------------------------------------===//
// arith.cmpf
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @cmpf
func.func @cmpf(%arg0 : f32, %arg1 : f32) {
  // CHECK: spirv.FOrdEqual
  %1 = arith.cmpf oeq, %arg0, %arg1 : f32
  return
}

// CHECK-LABEL: @vec1cmpf
func.func @vec1cmpf(%arg0 : vector<1xf32>, %arg1 : vector<1xf32>) {
  // CHECK: spirv.FOrdGreaterThan
  %0 = arith.cmpf ogt, %arg0, %arg1 : vector<1xf32>
  // CHECK: spirv.FUnordLessThan
  %1 = arith.cmpf ult, %arg0, %arg1 : vector<1xf32>
  return
}

// -----

//===----------------------------------------------------------------------===//
// arith.cmpi
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @cmpi
func.func @cmpi(%arg0 : i32, %arg1 : i32) {
  // CHECK: spirv.IEqual
  %0 = arith.cmpi eq, %arg0, %arg1 : i32
  return
}

// CHECK-LABEL: @indexcmpi
func.func @indexcmpi(%arg0 : index, %arg1 : index) {
  // CHECK: spirv.IEqual
  %0 = arith.cmpi eq, %arg0, %arg1 : index
  return
}

// CHECK-LABEL: @vec1cmpi
func.func @vec1cmpi(%arg0 : vector<1xi32>, %arg1 : vector<1xi32>) {
  // CHECK: spirv.ULessThan
  %0 = arith.cmpi ult, %arg0, %arg1 : vector<1xi32>
  // CHECK: spirv.SGreaterThan
  %1 = arith.cmpi sgt, %arg0, %arg1 : vector<1xi32>
  return
}

// CHECK-LABEL: @boolcmpi_equality
func.func @boolcmpi_equality(%arg0 : i1, %arg1 : i1) {
  // CHECK: spirv.LogicalEqual
  %0 = arith.cmpi eq, %arg0, %arg1 : i1
  // CHECK: spirv.LogicalNotEqual
  %1 = arith.cmpi ne, %arg0, %arg1 : i1
  return
}

// CHECK-LABEL: @boolcmpi_unsigned
func.func @boolcmpi_unsigned(%arg0 : i1, %arg1 : i1) {
  // CHECK-COUNT-2: spirv.Select
  // CHECK: spirv.UGreaterThanEqual
  %0 = arith.cmpi uge, %arg0, %arg1 : i1
  // CHECK-COUNT-2: spirv.Select
  // CHECK: spirv.ULessThan
  %1 = arith.cmpi ult, %arg0, %arg1 : i1
  return
}

// CHECK-LABEL: @vec1boolcmpi_equality
func.func @vec1boolcmpi_equality(%arg0 : vector<1xi1>, %arg1 : vector<1xi1>) {
  // CHECK: spirv.LogicalEqual
  %0 = arith.cmpi eq, %arg0, %arg1 : vector<1xi1>
  // CHECK: spirv.LogicalNotEqual
  %1 = arith.cmpi ne, %arg0, %arg1 : vector<1xi1>
  return
}

// CHECK-LABEL: @vec1boolcmpi_unsigned
func.func @vec1boolcmpi_unsigned(%arg0 : vector<1xi1>, %arg1 : vector<1xi1>) {
  // CHECK-COUNT-2: spirv.Select
  // CHECK: spirv.UGreaterThanEqual
  %0 = arith.cmpi uge, %arg0, %arg1 : vector<1xi1>
  // CHECK-COUNT-2: spirv.Select
  // CHECK: spirv.ULessThan
  %1 = arith.cmpi ult, %arg0, %arg1 : vector<1xi1>
  return
}

// CHECK-LABEL: @vecboolcmpi_equality
func.func @vecboolcmpi_equality(%arg0 : vector<4xi1>, %arg1 : vector<4xi1>) {
  // CHECK: spirv.LogicalEqual
  %0 = arith.cmpi eq, %arg0, %arg1 : vector<4xi1>
  // CHECK: spirv.LogicalNotEqual
  %1 = arith.cmpi ne, %arg0, %arg1 : vector<4xi1>
  return
}

// CHECK-LABEL: @vecboolcmpi_unsigned
func.func @vecboolcmpi_unsigned(%arg0 : vector<3xi1>, %arg1 : vector<3xi1>) {
  // CHECK-COUNT-2: spirv.Select
  // CHECK: spirv.UGreaterThanEqual
  %0 = arith.cmpi uge, %arg0, %arg1 : vector<3xi1>
  // CHECK-COUNT-2: spirv.Select
  // CHECK: spirv.ULessThan
  %1 = arith.cmpi ult, %arg0, %arg1 : vector<3xi1>
  return
}
