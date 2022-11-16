// RUN: mlir-opt -split-input-file -convert-arith-to-spirv -verify-diagnostics %s | FileCheck %s

//===----------------------------------------------------------------------===//
// arithmetic ops
//===----------------------------------------------------------------------===//

module attributes {
  spirv.target_env = #spirv.target_env<
    #spirv.vce<v1.0, [Int8, Int16, Int64, Float16, Float64, Shader], []>, #spirv.resource_limits<>>
} {

// Check integer operation conversions.
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

// CHECK-LABEL: @index_scalar
func.func @index_scalar(%lhs: index, %rhs: index) {
  // CHECK: spirv.IAdd %{{.*}}, %{{.*}}: i32
  %0 = arith.addi %lhs, %rhs: index
  // CHECK: spirv.ISub %{{.*}}, %{{.*}}: i32
  %1 = arith.subi %lhs, %rhs: index
  // CHECK: spirv.IMul %{{.*}}, %{{.*}}: i32
  %2 = arith.muli %lhs, %rhs: index
  // CHECK: spirv.SDiv %{{.*}}, %{{.*}}: i32
  %3 = arith.divsi %lhs, %rhs: index
  // CHECK: spirv.UDiv %{{.*}}, %{{.*}}: i32
  %4 = arith.divui %lhs, %rhs: index
  // CHECK: spirv.UMod %{{.*}}, %{{.*}}: i32
  %5 = arith.remui %lhs, %rhs: index
  return
}

// CHECK-LABEL: @index_scalar_srem
// CHECK-SAME: (%[[A:.+]]: index, %[[B:.+]]: index)
func.func @index_scalar_srem(%lhs: index, %rhs: index) {
  // CHECK: %[[LHS:.+]] = builtin.unrealized_conversion_cast %[[A]] : index to i32
  // CHECK: %[[RHS:.+]] = builtin.unrealized_conversion_cast %[[B]] : index to i32
  // CHECK: %[[LABS:.+]] = spirv.GL.SAbs %[[LHS]] : i32
  // CHECK: %[[RABS:.+]] = spirv.GL.SAbs %[[RHS]] : i32
  // CHECK:  %[[ABS:.+]] = spirv.UMod %[[LABS]], %[[RABS]] : i32
  // CHECK:  %[[POS:.+]] = spirv.IEqual %[[LHS]], %[[LABS]] : i32
  // CHECK:  %[[NEG:.+]] = spirv.SNegate %[[ABS]] : i32
  // CHECK:      %{{.+}} = spirv.Select %[[POS]], %[[ABS]], %[[NEG]] : i1, i32
  %0 = arith.remsi %lhs, %rhs: index
  return
}

// Check integer add-with-carry conversions.
// CHECK-LABEL: @int32_scalar_addui_carry
// CHECK-SAME: (%[[LHS:.+]]: i32, %[[RHS:.+]]: i32)
func.func @int32_scalar_addui_carry(%lhs: i32, %rhs: i32) -> (i32, i1) {
  // CHECK-NEXT: %[[IAC:.+]] = spirv.IAddCarry %[[LHS]], %[[RHS]] : !spirv.struct<(i32, i32)>
  // CHECK-DAG:  %[[SUM:.+]] = spirv.CompositeExtract %[[IAC]][0 : i32] : !spirv.struct<(i32, i32)>
  // CHECK-DAG:  %[[C0:.+]]  = spirv.CompositeExtract %[[IAC]][1 : i32] : !spirv.struct<(i32, i32)>
  // CHECK-DAG:  %[[ONE:.+]] = spirv.Constant 1 : i32
  // CHECK-NEXT: %[[C1:.+]]  = spirv.IEqual %[[C0]], %[[ONE]] : i32
  // CHECK-NEXT: return %[[SUM]], %[[C1]] : i32, i1
  %sum, %carry = arith.addui_carry %lhs, %rhs: i32, i1
  return %sum, %carry : i32, i1
}

// CHECK-LABEL: @int32_vector_addui_carry
// CHECK-SAME: (%[[LHS:.+]]: vector<4xi32>, %[[RHS:.+]]: vector<4xi32>)
func.func @int32_vector_addui_carry(%lhs: vector<4xi32>, %rhs: vector<4xi32>) -> (vector<4xi32>, vector<4xi1>) {
  // CHECK-NEXT: %[[IAC:.+]] = spirv.IAddCarry %[[LHS]], %[[RHS]] : !spirv.struct<(vector<4xi32>, vector<4xi32>)>
  // CHECK-DAG:  %[[SUM:.+]] = spirv.CompositeExtract %[[IAC]][0 : i32] : !spirv.struct<(vector<4xi32>, vector<4xi32>)>
  // CHECK-DAG:  %[[C0:.+]]  = spirv.CompositeExtract %[[IAC]][1 : i32] : !spirv.struct<(vector<4xi32>, vector<4xi32>)>
  // CHECK-DAG:  %[[ONE:.+]] = spirv.Constant dense<1> : vector<4xi32>
  // CHECK-NEXT: %[[C1:.+]]  = spirv.IEqual %[[C0]], %[[ONE]] : vector<4xi32>
  // CHECK-NEXT: return %[[SUM]], %[[C1]] : vector<4xi32>, vector<4xi1>
  %sum, %carry = arith.addui_carry %lhs, %rhs: vector<4xi32>, vector<4xi1>
  return %sum, %carry : vector<4xi32>, vector<4xi1>
}

// Check float unary operation conversions.
// CHECK-LABEL: @float32_unary_scalar
func.func @float32_unary_scalar(%arg0: f32) {
  // CHECK: spirv.FNegate %{{.*}}: f32
  %0 = arith.negf %arg0 : f32
  return
}

// Check float binary operation conversions.
// CHECK-LABEL: @float32_binary_scalar
func.func @float32_binary_scalar(%lhs: f32, %rhs: f32) {
  // CHECK: spirv.FAdd %{{.*}}, %{{.*}}: f32
  %0 = arith.addf %lhs, %rhs: f32
  // CHECK: spirv.FSub %{{.*}}, %{{.*}}: f32
  %1 = arith.subf %lhs, %rhs: f32
  // CHECK: spirv.FMul %{{.*}}, %{{.*}}: f32
  %2 = arith.mulf %lhs, %rhs: f32
  // CHECK: spirv.FDiv %{{.*}}, %{{.*}}: f32
  %3 = arith.divf %lhs, %rhs: f32
  // CHECK: spirv.FRem %{{.*}}, %{{.*}}: f32
  %4 = arith.remf %lhs, %rhs: f32
  return
}

// Check int vector types.
// CHECK-LABEL: @int_vector234
func.func @int_vector234(%arg0: vector<2xi8>, %arg1: vector<4xi64>) {
  // CHECK: spirv.SDiv %{{.*}}, %{{.*}}: vector<2xi8>
  %0 = arith.divsi %arg0, %arg0: vector<2xi8>
  // CHECK: spirv.UDiv %{{.*}}, %{{.*}}: vector<4xi64>
  %1 = arith.divui %arg1, %arg1: vector<4xi64>
  return
}

// CHECK-LABEL: @vector_srem
// CHECK-SAME: (%[[LHS:.+]]: vector<3xi16>, %[[RHS:.+]]: vector<3xi16>)
func.func @vector_srem(%arg0: vector<3xi16>, %arg1: vector<3xi16>) {
  // CHECK: %[[LABS:.+]] = spirv.GL.SAbs %[[LHS]] : vector<3xi16>
  // CHECK: %[[RABS:.+]] = spirv.GL.SAbs %[[RHS]] : vector<3xi16>
  // CHECK:  %[[ABS:.+]] = spirv.UMod %[[LABS]], %[[RABS]] : vector<3xi16>
  // CHECK:  %[[POS:.+]] = spirv.IEqual %[[LHS]], %[[LABS]] : vector<3xi16>
  // CHECK:  %[[NEG:.+]] = spirv.SNegate %[[ABS]] : vector<3xi16>
  // CHECK:      %{{.+}} = spirv.Select %[[POS]], %[[ABS]], %[[NEG]] : vector<3xi1>, vector<3xi16>
  %0 = arith.remsi %arg0, %arg1: vector<3xi16>
  return
}

// Check float vector types.
// CHECK-LABEL: @float_vector234
func.func @float_vector234(%arg0: vector<2xf16>, %arg1: vector<3xf64>) {
  // CHECK: spirv.FAdd %{{.*}}, %{{.*}}: vector<2xf16>
  %0 = arith.addf %arg0, %arg0: vector<2xf16>
  // CHECK: spirv.FMul %{{.*}}, %{{.*}}: vector<3xf64>
  %1 = arith.mulf %arg1, %arg1: vector<3xf64>
  return
}

// CHECK-LABEL: @one_elem_vector
func.func @one_elem_vector(%arg0: vector<1xi32>) {
  // CHECK: spirv.IAdd %{{.+}}, %{{.+}}: i32
  %0 = arith.addi %arg0, %arg0: vector<1xi32>
  return
}

} // end module

// -----

//===----------------------------------------------------------------------===//
// std bit ops
//===----------------------------------------------------------------------===//

module attributes {
  spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [], []>, #spirv.resource_limits<>>
} {

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

} // end module

// -----

//===----------------------------------------------------------------------===//
// arith.cmpf
//===----------------------------------------------------------------------===//

module attributes {
  spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [], []>, #spirv.resource_limits<>>
} {

// CHECK-LABEL: @cmpf
func.func @cmpf(%arg0 : f32, %arg1 : f32) {
  // CHECK: spirv.FOrdEqual
  %1 = arith.cmpf oeq, %arg0, %arg1 : f32
  // CHECK: spirv.FOrdGreaterThan
  %2 = arith.cmpf ogt, %arg0, %arg1 : f32
  // CHECK: spirv.FOrdGreaterThanEqual
  %3 = arith.cmpf oge, %arg0, %arg1 : f32
  // CHECK: spirv.FOrdLessThan
  %4 = arith.cmpf olt, %arg0, %arg1 : f32
  // CHECK: spirv.FOrdLessThanEqual
  %5 = arith.cmpf ole, %arg0, %arg1 : f32
  // CHECK: spirv.FOrdNotEqual
  %6 = arith.cmpf one, %arg0, %arg1 : f32
  // CHECK: spirv.FUnordEqual
  %7 = arith.cmpf ueq, %arg0, %arg1 : f32
  // CHECK: spirv.FUnordGreaterThan
  %8 = arith.cmpf ugt, %arg0, %arg1 : f32
  // CHECK: spirv.FUnordGreaterThanEqual
  %9 = arith.cmpf uge, %arg0, %arg1 : f32
  // CHECK: spirv.FUnordLessThan
  %10 = arith.cmpf ult, %arg0, %arg1 : f32
  // CHECK: FUnordLessThanEqual
  %11 = arith.cmpf ule, %arg0, %arg1 : f32
  // CHECK: spirv.FUnordNotEqual
  %12 = arith.cmpf une, %arg0, %arg1 : f32
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

} // end module

// -----

// With Kernel capability, we can convert NaN check to spirv.Ordered/spirv.Unordered.
module attributes {
  spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Kernel], []>, #spirv.resource_limits<>>
} {

// CHECK-LABEL: @cmpf
func.func @cmpf(%arg0 : f32, %arg1 : f32) {
  // CHECK: spirv.Ordered
  %0 = arith.cmpf ord, %arg0, %arg1 : f32
  // CHECK: spirv.Unordered
  %1 = arith.cmpf uno, %arg0, %arg1 : f32
  return
}

} // end module

// -----

// Without Kernel capability, we need to convert NaN check to spirv.IsNan.
module attributes {
  spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [], []>, #spirv.resource_limits<>>
} {

// CHECK-LABEL: @cmpf
// CHECK-SAME: %[[LHS:.+]]: f32, %[[RHS:.+]]: f32
func.func @cmpf(%arg0 : f32, %arg1 : f32) {
  // CHECK:      %[[LHS_NAN:.+]] = spirv.IsNan %[[LHS]] : f32
  // CHECK-NEXT: %[[RHS_NAN:.+]] = spirv.IsNan %[[RHS]] : f32
  // CHECK-NEXT: %[[OR:.+]] = spirv.LogicalOr %[[LHS_NAN]], %[[RHS_NAN]] : i1
  // CHECK-NEXT: %{{.+}} = spirv.LogicalNot %[[OR]] : i1
  %0 = arith.cmpf ord, %arg0, %arg1 : f32

  // CHECK-NEXT: %[[LHS_NAN:.+]] = spirv.IsNan %[[LHS]] : f32
  // CHECK-NEXT: %[[RHS_NAN:.+]] = spirv.IsNan %[[RHS]] : f32
  // CHECK-NEXT: %{{.+}} = spirv.LogicalOr %[[LHS_NAN]], %[[RHS_NAN]] : i1
  %1 = arith.cmpf uno, %arg0, %arg1 : f32
  return
}

} // end module

// -----

//===----------------------------------------------------------------------===//
// arith.cmpi
//===----------------------------------------------------------------------===//

module attributes {
  spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [], []>, #spirv.resource_limits<>>
} {

// CHECK-LABEL: @cmpi
func.func @cmpi(%arg0 : i32, %arg1 : i32) {
  // CHECK: spirv.IEqual
  %0 = arith.cmpi eq, %arg0, %arg1 : i32
  // CHECK: spirv.INotEqual
  %1 = arith.cmpi ne, %arg0, %arg1 : i32
  // CHECK: spirv.SLessThan
  %2 = arith.cmpi slt, %arg0, %arg1 : i32
  // CHECK: spirv.SLessThanEqual
  %3 = arith.cmpi sle, %arg0, %arg1 : i32
  // CHECK: spirv.SGreaterThan
  %4 = arith.cmpi sgt, %arg0, %arg1 : i32
  // CHECK: spirv.SGreaterThanEqual
  %5 = arith.cmpi sge, %arg0, %arg1 : i32
  // CHECK: spirv.ULessThan
  %6 = arith.cmpi ult, %arg0, %arg1 : i32
  // CHECK: spirv.ULessThanEqual
  %7 = arith.cmpi ule, %arg0, %arg1 : i32
  // CHECK: spirv.UGreaterThan
  %8 = arith.cmpi ugt, %arg0, %arg1 : i32
  // CHECK: spirv.UGreaterThanEqual
  %9 = arith.cmpi uge, %arg0, %arg1 : i32
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


} // end module

// -----

//===----------------------------------------------------------------------===//
// arith.constant
//===----------------------------------------------------------------------===//

module attributes {
  spirv.target_env = #spirv.target_env<
    #spirv.vce<v1.0, [Int8, Int16, Int64, Float16, Float64], []>, #spirv.resource_limits<>>
} {

// CHECK-LABEL: @constant
func.func @constant() {
  // CHECK: spirv.Constant true
  %0 = arith.constant true
  // CHECK: spirv.Constant 42 : i32
  %1 = arith.constant 42 : i32
  // CHECK: spirv.Constant 5.000000e-01 : f32
  %2 = arith.constant 0.5 : f32
  // CHECK: spirv.Constant dense<[2, 3]> : vector<2xi32>
  %3 = arith.constant dense<[2, 3]> : vector<2xi32>
  // CHECK: spirv.Constant 1 : i32
  %4 = arith.constant 1 : index
  // CHECK: spirv.Constant dense<1> : tensor<6xi32> : !spirv.array<6 x i32>
  %5 = arith.constant dense<1> : tensor<2x3xi32>
  // CHECK: spirv.Constant dense<1.000000e+00> : tensor<6xf32> : !spirv.array<6 x f32>
  %6 = arith.constant dense<1.0> : tensor<2x3xf32>
  // CHECK: spirv.Constant dense<{{\[}}1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00]> : tensor<6xf32> : !spirv.array<6 x f32>
  %7 = arith.constant dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf32>
  // CHECK: spirv.Constant dense<{{\[}}1, 2, 3, 4, 5, 6]> : tensor<6xi32> : !spirv.array<6 x i32>
  %8 = arith.constant dense<[[1, 2, 3], [4, 5, 6]]> : tensor<2x3xi32>
  // CHECK: spirv.Constant dense<{{\[}}1, 2, 3, 4, 5, 6]> : tensor<6xi32> : !spirv.array<6 x i32>
  %9 = arith.constant dense<[[1, 2], [3, 4], [5, 6]]> : tensor<3x2xi32>
  // CHECK: spirv.Constant dense<{{\[}}1, 2, 3, 4, 5, 6]> : tensor<6xi32> : !spirv.array<6 x i32>
  %10 = arith.constant dense<[1, 2, 3, 4, 5, 6]> : tensor<6xi32>
  return
}

// CHECK-LABEL: @constant_16bit
func.func @constant_16bit() {
  // CHECK: spirv.Constant 4 : i16
  %0 = arith.constant 4 : i16
  // CHECK: spirv.Constant 5.000000e+00 : f16
  %1 = arith.constant 5.0 : f16
  // CHECK: spirv.Constant dense<[2, 3]> : vector<2xi16>
  %2 = arith.constant dense<[2, 3]> : vector<2xi16>
  // CHECK: spirv.Constant dense<4.000000e+00> : tensor<5xf16> : !spirv.array<5 x f16>
  %3 = arith.constant dense<4.0> : tensor<5xf16>
  return
}

// CHECK-LABEL: @constant_64bit
func.func @constant_64bit() {
  // CHECK: spirv.Constant 4 : i64
  %0 = arith.constant 4 : i64
  // CHECK: spirv.Constant 5.000000e+00 : f64
  %1 = arith.constant 5.0 : f64
  // CHECK: spirv.Constant dense<[2, 3]> : vector<2xi64>
  %2 = arith.constant dense<[2, 3]> : vector<2xi64>
  // CHECK: spirv.Constant dense<4.000000e+00> : tensor<5xf64> : !spirv.array<5 x f64>
  %3 = arith.constant dense<4.0> : tensor<5xf64>
  return
}

// CHECK-LABEL: @constant_size1
func.func @constant_size1() {
  // CHECK: spirv.Constant true
  %0 = arith.constant dense<true> : tensor<1xi1>
  // CHECK: spirv.Constant 4 : i64
  %1 = arith.constant dense<4> : vector<1xi64>
  // CHECK: spirv.Constant 5.000000e+00 : f64
  %2 = arith.constant dense<5.0> : tensor<1xf64>
  return
}

} // end module

// -----

// Check that constants are widened to 32-bit when no special capability.
module attributes {
  spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [], []>, #spirv.resource_limits<>>
} {

// CHECK-LABEL: @constant_16bit
func.func @constant_16bit() {
  // CHECK: spirv.Constant 4 : i32
  %0 = arith.constant 4 : i16
  // CHECK: spirv.Constant 5.000000e+00 : f32
  %1 = arith.constant 5.0 : f16
  // CHECK: spirv.Constant dense<[2, 3]> : vector<2xi32>
  %2 = arith.constant dense<[2, 3]> : vector<2xi16>
  // CHECK: spirv.Constant dense<4.000000e+00> : tensor<5xf32> : !spirv.array<5 x f32>
  %3 = arith.constant dense<4.0> : tensor<5xf16>
  // CHECK: spirv.Constant dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00]> : tensor<4xf32> : !spirv.array<4 x f32>
  %4 = arith.constant dense<[[1.0, 2.0], [3.0, 4.0]]> : tensor<2x2xf16>
  return
}

// CHECK-LABEL: @constant_size1
func.func @constant_size1() {
  // CHECK: spirv.Constant 4 : i32
  %0 = arith.constant dense<4> : vector<1xi16>
  // CHECK: spirv.Constant 5.000000e+00 : f32
  %1 = arith.constant dense<5.0> : tensor<1xf16>
  return
}

// CHECK-LABEL: @corner_cases
func.func @corner_cases() {
 // CHECK: %{{.*}} = spirv.Constant -1 : i32
  %5 = arith.constant -1 : i16
  // CHECK: %{{.*}} = spirv.Constant -2 : i32
  %6 = arith.constant -2 : i16
  // CHECK: %{{.*}} = spirv.Constant -1 : i32
  %7 = arith.constant -1 : index
  // CHECK: %{{.*}} = spirv.Constant -2 : i32
  %8 = arith.constant -2 : index

  // CHECK: spirv.Constant false
  %9 = arith.constant false
  // CHECK: spirv.Constant true
  %10 = arith.constant true

  return
}

} // end module

// -----

//===----------------------------------------------------------------------===//
// std cast ops
//===----------------------------------------------------------------------===//

module attributes {
  spirv.target_env = #spirv.target_env<
    #spirv.vce<v1.0, [Int8, Int16, Int64, Float16, Float64], []>, #spirv.resource_limits<>>
} {

// CHECK-LABEL: index_cast1
func.func @index_cast1(%arg0: i16) {
  // CHECK: spirv.SConvert %{{.+}} : i16 to i32
  %0 = arith.index_cast %arg0 : i16 to index
  return
}

// CHECK-LABEL: index_cast2
func.func @index_cast2(%arg0: index) {
  // CHECK: spirv.SConvert %{{.+}} : i32 to i16
  %0 = arith.index_cast %arg0 : index to i16
  return
}

// CHECK-LABEL: index_cast3
func.func @index_cast3(%arg0: i32) {
  // CHECK-NOT: spirv.SConvert
  %0 = arith.index_cast %arg0 : i32 to index
  return
}

// CHECK-LABEL: index_cast4
func.func @index_cast4(%arg0: index) {
  // CHECK-NOT: spirv.UConvert
  %0 = arith.index_cast %arg0 : index to i32
  return
}

// CHECK-LABEL: index_castui1
func.func @index_castui1(%arg0: i16) {
  // CHECK: spirv.UConvert %{{.+}} : i16 to i32
  %0 = arith.index_castui %arg0 : i16 to index
  return
}

// CHECK-LABEL: index_castui2
func.func @index_castui2(%arg0: index) {
  // CHECK: spirv.UConvert %{{.+}} : i32 to i16
  %0 = arith.index_castui %arg0 : index to i16
  return
}

// CHECK-LABEL: index_castui3
func.func @index_castui3(%arg0: i32) {
  // CHECK-NOT: spirv.UConvert
  %0 = arith.index_castui %arg0 : i32 to index
  return
}

// CHECK-LABEL: index_castui4
func.func @index_castui4(%arg0: index) {
  // CHECK-NOT: spirv.UConvert
  %0 = arith.index_cast %arg0 : index to i32
  return
}

// CHECK-LABEL: @bit_cast
func.func @bit_cast(%arg0: vector<2xf32>, %arg1: i64) {
  // CHECK: spirv.Bitcast %{{.+}} : vector<2xf32> to vector<2xi32>
  %0 = arith.bitcast %arg0 : vector<2xf32> to vector<2xi32>
  // CHECK: spirv.Bitcast %{{.+}} : i64 to f64
  %1 = arith.bitcast %arg1 : i64 to f64
  return
}

// CHECK-LABEL: @fpext1
func.func @fpext1(%arg0: f16) -> f64 {
  // CHECK: spirv.FConvert %{{.*}} : f16 to f64
  %0 = arith.extf %arg0 : f16 to f64
  return %0 : f64
}

// CHECK-LABEL: @fpext2
func.func @fpext2(%arg0 : f32) -> f64 {
  // CHECK: spirv.FConvert %{{.*}} : f32 to f64
  %0 = arith.extf %arg0 : f32 to f64
  return %0 : f64
}

// CHECK-LABEL: @fptrunc1
func.func @fptrunc1(%arg0 : f64) -> f16 {
  // CHECK: spirv.FConvert %{{.*}} : f64 to f16
  %0 = arith.truncf %arg0 : f64 to f16
  return %0 : f16
}

// CHECK-LABEL: @fptrunc2
func.func @fptrunc2(%arg0: f32) -> f16 {
  // CHECK: spirv.FConvert %{{.*}} : f32 to f16
  %0 = arith.truncf %arg0 : f32 to f16
  return %0 : f16
}

// CHECK-LABEL: @sitofp1
func.func @sitofp1(%arg0 : i32) -> f32 {
  // CHECK: spirv.ConvertSToF %{{.*}} : i32 to f32
  %0 = arith.sitofp %arg0 : i32 to f32
  return %0 : f32
}

// CHECK-LABEL: @sitofp2
func.func @sitofp2(%arg0 : i64) -> f64 {
  // CHECK: spirv.ConvertSToF %{{.*}} : i64 to f64
  %0 = arith.sitofp %arg0 : i64 to f64
  return %0 : f64
}

// CHECK-LABEL: @uitofp_i16_f32
func.func @uitofp_i16_f32(%arg0: i16) -> f32 {
  // CHECK: spirv.ConvertUToF %{{.*}} : i16 to f32
  %0 = arith.uitofp %arg0 : i16 to f32
  return %0 : f32
}

// CHECK-LABEL: @uitofp_i32_f32
func.func @uitofp_i32_f32(%arg0 : i32) -> f32 {
  // CHECK: spirv.ConvertUToF %{{.*}} : i32 to f32
  %0 = arith.uitofp %arg0 : i32 to f32
  return %0 : f32
}

// CHECK-LABEL: @uitofp_i1_f32
func.func @uitofp_i1_f32(%arg0 : i1) -> f32 {
  // CHECK: %[[ZERO:.+]] = spirv.Constant 0.000000e+00 : f32
  // CHECK: %[[ONE:.+]] = spirv.Constant 1.000000e+00 : f32
  // CHECK: spirv.Select %{{.*}}, %[[ONE]], %[[ZERO]] : i1, f32
  %0 = arith.uitofp %arg0 : i1 to f32
  return %0 : f32
}

// CHECK-LABEL: @uitofp_i1_f64
func.func @uitofp_i1_f64(%arg0 : i1) -> f64 {
  // CHECK: %[[ZERO:.+]] = spirv.Constant 0.000000e+00 : f64
  // CHECK: %[[ONE:.+]] = spirv.Constant 1.000000e+00 : f64
  // CHECK: spirv.Select %{{.*}}, %[[ONE]], %[[ZERO]] : i1, f64
  %0 = arith.uitofp %arg0 : i1 to f64
  return %0 : f64
}

// CHECK-LABEL: @uitofp_vec_i1_f32
func.func @uitofp_vec_i1_f32(%arg0 : vector<4xi1>) -> vector<4xf32> {
  // CHECK: %[[ZERO:.+]] = spirv.Constant dense<0.000000e+00> : vector<4xf32>
  // CHECK: %[[ONE:.+]] = spirv.Constant dense<1.000000e+00> : vector<4xf32>
  // CHECK: spirv.Select %{{.*}}, %[[ONE]], %[[ZERO]] : vector<4xi1>, vector<4xf32>
  %0 = arith.uitofp %arg0 : vector<4xi1> to vector<4xf32>
  return %0 : vector<4xf32>
}

// CHECK-LABEL: @uitofp_vec_i1_f64
spirv.func @uitofp_vec_i1_f64(%arg0: vector<4xi1>) -> vector<4xf64> "None" {
  // CHECK: %[[ZERO:.+]] = spirv.Constant dense<0.000000e+00> : vector<4xf64>
  // CHECK: %[[ONE:.+]] = spirv.Constant dense<1.000000e+00> : vector<4xf64>
  // CHECK: spirv.Select %{{.*}}, %[[ONE]], %[[ZERO]] : vector<4xi1>, vector<4xf64>
  %0 = spirv.Constant dense<0.000000e+00> : vector<4xf64>
  %1 = spirv.Constant dense<1.000000e+00> : vector<4xf64>
  %2 = spirv.Select %arg0, %1, %0 : vector<4xi1>, vector<4xf64>
  spirv.ReturnValue %2 : vector<4xf64>
}

// CHECK-LABEL: @sexti1
func.func @sexti1(%arg0: i16) -> i64 {
  // CHECK: spirv.SConvert %{{.*}} : i16 to i64
  %0 = arith.extsi %arg0 : i16 to i64
  return %0 : i64
}

// CHECK-LABEL: @sexti2
func.func @sexti2(%arg0 : i32) -> i64 {
  // CHECK: spirv.SConvert %{{.*}} : i32 to i64
  %0 = arith.extsi %arg0 : i32 to i64
  return %0 : i64
}

// CHECK-LABEL: @sext_bool_scalar
// CHECK-SAME:  ([[ARG:%.+]]: i1) -> i32
func.func @sext_bool_scalar(%arg0 : i1) -> i32 {
  // CHECK-DAG:  [[ONES:%.+]] = spirv.Constant -1 : i32
  // CHECK-DAG:  [[ZERO:%.+]] = spirv.Constant 0 : i32
  // CHECK:      [[SEL:%.+]]  = spirv.Select [[ARG]], [[ONES]], [[ZERO]] : i1, i32
  // CHECK-NEXT: return [[SEL]] : i32
  %0 = arith.extsi %arg0 : i1 to i32
  return %0 : i32
}

// CHECK-LABEL: @sext_bool_vector
// CHECK-SAME:  ([[ARG:%.+]]: vector<3xi1>) -> vector<3xi32>
func.func @sext_bool_vector(%arg0 : vector<3xi1>) -> vector<3xi32> {
  // CHECK-DAG:  [[ONES:%.+]] = spirv.Constant dense<-1> : vector<3xi32>
  // CHECK-DAG:  [[ZERO:%.+]] = spirv.Constant dense<0> : vector<3xi32>
  // CHECK:      [[SEL:%.+]]  = spirv.Select [[ARG]], [[ONES]], [[ZERO]] : vector<3xi1>, vector<3xi32>
  // CHECK-NEXT: return [[SEL]] : vector<3xi32>
  %0 = arith.extsi %arg0 : vector<3xi1> to vector<3xi32>
  return %0 : vector<3xi32>
}

// CHECK-LABEL: @zexti1
func.func @zexti1(%arg0: i16) -> i64 {
  // CHECK: spirv.UConvert %{{.*}} : i16 to i64
  %0 = arith.extui %arg0 : i16 to i64
  return %0 : i64
}

// CHECK-LABEL: @zexti2
func.func @zexti2(%arg0 : i32) -> i64 {
  // CHECK: spirv.UConvert %{{.*}} : i32 to i64
  %0 = arith.extui %arg0 : i32 to i64
  return %0 : i64
}

// CHECK-LABEL: @zexti3
func.func @zexti3(%arg0 : i1) -> i32 {
  // CHECK: %[[ZERO:.+]] = spirv.Constant 0 : i32
  // CHECK: %[[ONE:.+]] = spirv.Constant 1 : i32
  // CHECK: spirv.Select %{{.*}}, %[[ONE]], %[[ZERO]] : i1, i32
  %0 = arith.extui %arg0 : i1 to i32
  return %0 : i32
}

// CHECK-LABEL: @zexti4
func.func @zexti4(%arg0 : vector<4xi1>) -> vector<4xi32> {
  // CHECK: %[[ZERO:.+]] = spirv.Constant dense<0> : vector<4xi32>
  // CHECK: %[[ONE:.+]] = spirv.Constant dense<1> : vector<4xi32>
  // CHECK: spirv.Select %{{.*}}, %[[ONE]], %[[ZERO]] : vector<4xi1>, vector<4xi32>
  %0 = arith.extui %arg0 : vector<4xi1> to vector<4xi32>
  return %0 : vector<4xi32>
}

// CHECK-LABEL: @zexti5
func.func @zexti5(%arg0 : vector<4xi1>) -> vector<4xi64> {
  // CHECK: %[[ZERO:.+]] = spirv.Constant dense<0> : vector<4xi64>
  // CHECK: %[[ONE:.+]] = spirv.Constant dense<1> : vector<4xi64>
  // CHECK: spirv.Select %{{.*}}, %[[ONE]], %[[ZERO]] : vector<4xi1>, vector<4xi64>
  %0 = arith.extui %arg0 : vector<4xi1> to vector<4xi64>
  return %0 : vector<4xi64>
}

// CHECK-LABEL: @trunci1
func.func @trunci1(%arg0 : i64) -> i16 {
  // CHECK: spirv.SConvert %{{.*}} : i64 to i16
  %0 = arith.trunci %arg0 : i64 to i16
  return %0 : i16
}

// CHECK-LABEL: @trunci2
func.func @trunci2(%arg0: i32) -> i16 {
  // CHECK: spirv.SConvert %{{.*}} : i32 to i16
  %0 = arith.trunci %arg0 : i32 to i16
  return %0 : i16
}

// CHECK-LABEL: @trunc_to_i1
func.func @trunc_to_i1(%arg0: i32) -> i1 {
  // CHECK: %[[MASK:.*]] = spirv.Constant 1 : i32
  // CHECK: %[[MASKED_SRC:.*]] = spirv.BitwiseAnd %{{.*}}, %[[MASK]] : i32
  // CHECK: %[[IS_ONE:.*]] = spirv.IEqual %[[MASKED_SRC]], %[[MASK]] : i32
  // CHECK-DAG: %[[TRUE:.*]] = spirv.Constant true
  // CHECK-DAG: %[[FALSE:.*]] = spirv.Constant false
  // CHECK: spirv.Select %[[IS_ONE]], %[[TRUE]], %[[FALSE]] : i1, i1
  %0 = arith.trunci %arg0 : i32 to i1
  return %0 : i1
}

// CHECK-LABEL: @trunc_to_veci1
func.func @trunc_to_veci1(%arg0: vector<4xi32>) -> vector<4xi1> {
  // CHECK: %[[MASK:.*]] = spirv.Constant dense<1> : vector<4xi32>
  // CHECK: %[[MASKED_SRC:.*]] = spirv.BitwiseAnd %{{.*}}, %[[MASK]] : vector<4xi32>
  // CHECK: %[[IS_ONE:.*]] = spirv.IEqual %[[MASKED_SRC]], %[[MASK]] : vector<4xi32>
  // CHECK-DAG: %[[TRUE:.*]] = spirv.Constant dense<true> : vector<4xi1>
  // CHECK-DAG: %[[FALSE:.*]] = spirv.Constant dense<false> : vector<4xi1>
  // CHECK: spirv.Select %[[IS_ONE]], %[[TRUE]], %[[FALSE]] : vector<4xi1>, vector<4xi1>
  %0 = arith.trunci %arg0 : vector<4xi32> to vector<4xi1>
  return %0 : vector<4xi1>
}

// CHECK-LABEL: @fptosi1
func.func @fptosi1(%arg0 : f32) -> i32 {
  // CHECK: spirv.ConvertFToS %{{.*}} : f32 to i32
  %0 = arith.fptosi %arg0 : f32 to i32
  return %0 : i32
}

// CHECK-LABEL: @fptosi2
func.func @fptosi2(%arg0 : f16) -> i16 {
  // CHECK: spirv.ConvertFToS %{{.*}} : f16 to i16
  %0 = arith.fptosi %arg0 : f16 to i16
  return %0 : i16
}

} // end module

// -----

// Checks that cast types will be adjusted when missing special capabilities for
// certain non-32-bit scalar types.
module attributes {
  spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Float64], []>, #spirv.resource_limits<>>
} {

// CHECK-LABEL: @fpext1
// CHECK-SAME: %[[A:.*]]: f16
func.func @fpext1(%arg0: f16) -> f64 {
  // CHECK: %[[ARG:.+]] = builtin.unrealized_conversion_cast %[[A]] : f16 to f32
  // CHECK-NEXT: spirv.FConvert %[[ARG]] : f32 to f64
  %0 = arith.extf %arg0 : f16 to f64
  return %0: f64
}

// CHECK-LABEL: @fpext2
// CHECK-SAME: %[[ARG:.*]]: f32
func.func @fpext2(%arg0 : f32) -> f64 {
  // CHECK-NEXT: spirv.FConvert %[[ARG]] : f32 to f64
  %0 = arith.extf %arg0 : f32 to f64
  return %0: f64
}

} // end module

// -----

// Checks that cast types will be adjusted when missing special capabilities for
// certain non-32-bit scalar types.
module attributes {
  spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Float16], []>, #spirv.resource_limits<>>
} {

// CHECK-LABEL: @fptrunc1
// CHECK-SAME: %[[ARG:.*]]: f32
func.func @fptrunc1(%arg0: f32) -> f16 {
  // CHECK-NEXT: spirv.FConvert %[[ARG]] : f32 to f16
  %0 = arith.truncf %arg0 : f32 to f16
  return %0: f16
}

} // end module

// -----

// Check various lowerings for OpenCL.
module attributes {
  spirv.target_env = #spirv.target_env<
    #spirv.vce<v1.0, [Int16, Kernel], []>, #spirv.resource_limits<>>
} {

// Check integer operation conversions.
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
  // CHECK: spirv.CL.s_max %{{.*}}, %{{.*}}: i32
  %6 = arith.maxsi %lhs, %rhs : i32
  // CHECK: spirv.CL.u_max %{{.*}}, %{{.*}}: i32
  %7 = arith.maxui %lhs, %rhs : i32
  // CHECK: spirv.CL.s_min %{{.*}}, %{{.*}}: i32
  %8 = arith.minsi %lhs, %rhs : i32
  // CHECK: spirv.CL.u_min %{{.*}}, %{{.*}}: i32
  %9 = arith.minui %lhs, %rhs : i32
  return
}

// Check float binary operation conversions.
// CHECK-LABEL: @float32_binary_scalar
func.func @float32_binary_scalar(%lhs: f32, %rhs: f32) {
  // CHECK: spirv.FAdd %{{.*}}, %{{.*}}: f32
  %0 = arith.addf %lhs, %rhs: f32
  // CHECK: spirv.FSub %{{.*}}, %{{.*}}: f32
  %1 = arith.subf %lhs, %rhs: f32
  // CHECK: spirv.FMul %{{.*}}, %{{.*}}: f32
  %2 = arith.mulf %lhs, %rhs: f32
  // CHECK: spirv.FDiv %{{.*}}, %{{.*}}: f32
  %3 = arith.divf %lhs, %rhs: f32
  // CHECK: spirv.FRem %{{.*}}, %{{.*}}: f32
  %4 = arith.remf %lhs, %rhs: f32
  return
}

// CHECK-LABEL: @float32_minf_scalar
// CHECK-SAME: %[[LHS:.+]]: f32, %[[RHS:.+]]: f32
func.func @float32_minf_scalar(%arg0 : f32, %arg1 : f32) -> f32 {
  // CHECK: %[[MIN:.+]] = spirv.CL.fmin %arg0, %arg1 : f32
  // CHECK: %[[LHS_NAN:.+]] = spirv.IsNan %[[LHS]] : f32
  // CHECK: %[[RHS_NAN:.+]] = spirv.IsNan %[[RHS]] : f32
  // CHECK: %[[SELECT1:.+]] = spirv.Select %[[LHS_NAN]], %[[LHS]], %[[MIN]]
  // CHECK: %[[SELECT2:.+]] = spirv.Select %[[RHS_NAN]], %[[RHS]], %[[SELECT1]]
  %0 = arith.minf %arg0, %arg1 : f32
  // CHECK: return %[[SELECT2]]
  return %0: f32
}

// CHECK-LABEL: @float32_maxf_scalar
// CHECK-SAME: %[[LHS:.+]]: vector<2xf32>, %[[RHS:.+]]: vector<2xf32>
func.func @float32_maxf_scalar(%arg0 : vector<2xf32>, %arg1 : vector<2xf32>) -> vector<2xf32> {
  // CHECK: %[[MAX:.+]] = spirv.CL.fmax %arg0, %arg1 : vector<2xf32>
  // CHECK: %[[LHS_NAN:.+]] = spirv.IsNan %[[LHS]] : vector<2xf32>
  // CHECK: %[[RHS_NAN:.+]] = spirv.IsNan %[[RHS]] : vector<2xf32>
  // CHECK: %[[SELECT1:.+]] = spirv.Select %[[LHS_NAN]], %[[LHS]], %[[MAX]]
  // CHECK: %[[SELECT2:.+]] = spirv.Select %[[RHS_NAN]], %[[RHS]], %[[SELECT1]]
  %0 = arith.maxf %arg0, %arg1 : vector<2xf32>
  // CHECK: return %[[SELECT2]]
  return %0: vector<2xf32>
}

// CHECK-LABEL: @scalar_srem
// CHECK-SAME: (%[[LHS:.+]]: i32, %[[RHS:.+]]: i32)
func.func @scalar_srem(%lhs: i32, %rhs: i32) {
  // CHECK: %[[LABS:.+]] = spirv.CL.s_abs %[[LHS]] : i32
  // CHECK: %[[RABS:.+]] = spirv.CL.s_abs %[[RHS]] : i32
  // CHECK:  %[[ABS:.+]] = spirv.UMod %[[LABS]], %[[RABS]] : i32
  // CHECK:  %[[POS:.+]] = spirv.IEqual %[[LHS]], %[[LABS]] : i32
  // CHECK:  %[[NEG:.+]] = spirv.SNegate %[[ABS]] : i32
  // CHECK:      %{{.+}} = spirv.Select %[[POS]], %[[ABS]], %[[NEG]] : i1, i32
  %0 = arith.remsi %lhs, %rhs: i32
  return
}

// CHECK-LABEL: @vector_srem
// CHECK-SAME: (%[[LHS:.+]]: vector<3xi16>, %[[RHS:.+]]: vector<3xi16>)
func.func @vector_srem(%arg0: vector<3xi16>, %arg1: vector<3xi16>) {
  // CHECK: %[[LABS:.+]] = spirv.CL.s_abs %[[LHS]] : vector<3xi16>
  // CHECK: %[[RABS:.+]] = spirv.CL.s_abs %[[RHS]] : vector<3xi16>
  // CHECK:  %[[ABS:.+]] = spirv.UMod %[[LABS]], %[[RABS]] : vector<3xi16>
  // CHECK:  %[[POS:.+]] = spirv.IEqual %[[LHS]], %[[LABS]] : vector<3xi16>
  // CHECK:  %[[NEG:.+]] = spirv.SNegate %[[ABS]] : vector<3xi16>
  // CHECK:      %{{.+}} = spirv.Select %[[POS]], %[[ABS]], %[[NEG]] : vector<3xi1>, vector<3xi16>
  %0 = arith.remsi %arg0, %arg1: vector<3xi16>
  return
}

} // end module

// -----

module attributes {
  spirv.target_env = #spirv.target_env<
    #spirv.vce<v1.0, [Shader, Int8, Int16, Int64, Float16, Float64],
             [SPV_KHR_storage_buffer_storage_class]>, #spirv.resource_limits<>>
} {

// CHECK-LABEL: @select
func.func @select(%arg0 : i32, %arg1 : i32) {
  %0 = arith.cmpi sle, %arg0, %arg1 : i32
  // CHECK: spirv.Select
  %1 = arith.select %0, %arg0, %arg1 : i32
  return
}

} // end module

// -----

//===----------------------------------------------------------------------===//
// arith.select
//===----------------------------------------------------------------------===//

module attributes {
  spirv.target_env = #spirv.target_env<
    #spirv.vce<v1.0, [Int8, Int16, Int64, Float16, Float64, Shader], []>, #spirv.resource_limits<>>
} {

// Check integer operation conversions.
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
  // CHECK: spirv.GL.SMax %{{.*}}, %{{.*}}: i32
  %6 = arith.maxsi %lhs, %rhs : i32
  // CHECK: spirv.GL.UMax %{{.*}}, %{{.*}}: i32
  %7 = arith.maxui %lhs, %rhs : i32
  // CHECK: spirv.GL.SMin %{{.*}}, %{{.*}}: i32
  %8 = arith.minsi %lhs, %rhs : i32
  // CHECK: spirv.GL.UMin %{{.*}}, %{{.*}}: i32
  %9 = arith.minui %lhs, %rhs : i32
  return
}

// CHECK-LABEL: @scalar_srem
// CHECK-SAME: (%[[LHS:.+]]: i32, %[[RHS:.+]]: i32)
func.func @scalar_srem(%lhs: i32, %rhs: i32) {
  // CHECK: %[[LABS:.+]] = spirv.GL.SAbs %[[LHS]] : i32
  // CHECK: %[[RABS:.+]] = spirv.GL.SAbs %[[RHS]] : i32
  // CHECK:  %[[ABS:.+]] = spirv.UMod %[[LABS]], %[[RABS]] : i32
  // CHECK:  %[[POS:.+]] = spirv.IEqual %[[LHS]], %[[LABS]] : i32
  // CHECK:  %[[NEG:.+]] = spirv.SNegate %[[ABS]] : i32
  // CHECK:      %{{.+}} = spirv.Select %[[POS]], %[[ABS]], %[[NEG]] : i1, i32
  %0 = arith.remsi %lhs, %rhs: i32
  return
}

// Check float unary operation conversions.
// CHECK-LABEL: @float32_unary_scalar
func.func @float32_unary_scalar(%arg0: f32) {
  // CHECK: spirv.FNegate %{{.*}}: f32
  %5 = arith.negf %arg0 : f32
  return
}

// Check float binary operation conversions.
// CHECK-LABEL: @float32_binary_scalar
func.func @float32_binary_scalar(%lhs: f32, %rhs: f32) {
  // CHECK: spirv.FAdd %{{.*}}, %{{.*}}: f32
  %0 = arith.addf %lhs, %rhs: f32
  // CHECK: spirv.FSub %{{.*}}, %{{.*}}: f32
  %1 = arith.subf %lhs, %rhs: f32
  // CHECK: spirv.FMul %{{.*}}, %{{.*}}: f32
  %2 = arith.mulf %lhs, %rhs: f32
  // CHECK: spirv.FDiv %{{.*}}, %{{.*}}: f32
  %3 = arith.divf %lhs, %rhs: f32
  // CHECK: spirv.FRem %{{.*}}, %{{.*}}: f32
  %4 = arith.remf %lhs, %rhs: f32
  return
}

// CHECK-LABEL: @float32_minf_scalar
// CHECK-SAME: %[[LHS:.+]]: f32, %[[RHS:.+]]: f32
func.func @float32_minf_scalar(%arg0 : f32, %arg1 : f32) -> f32 {
  // CHECK: %[[MIN:.+]] = spirv.GL.FMin %arg0, %arg1 : f32
  // CHECK: %[[LHS_NAN:.+]] = spirv.IsNan %[[LHS]] : f32
  // CHECK: %[[RHS_NAN:.+]] = spirv.IsNan %[[RHS]] : f32
  // CHECK: %[[SELECT1:.+]] = spirv.Select %[[LHS_NAN]], %[[LHS]], %[[MIN]]
  // CHECK: %[[SELECT2:.+]] = spirv.Select %[[RHS_NAN]], %[[RHS]], %[[SELECT1]]
  %0 = arith.minf %arg0, %arg1 : f32
  // CHECK: return %[[SELECT2]]
  return %0: f32
}

// CHECK-LABEL: @float32_maxf_scalar
// CHECK-SAME: %[[LHS:.+]]: vector<2xf32>, %[[RHS:.+]]: vector<2xf32>
func.func @float32_maxf_scalar(%arg0 : vector<2xf32>, %arg1 : vector<2xf32>) -> vector<2xf32> {
  // CHECK: %[[MAX:.+]] = spirv.GL.FMax %arg0, %arg1 : vector<2xf32>
  // CHECK: %[[LHS_NAN:.+]] = spirv.IsNan %[[LHS]] : vector<2xf32>
  // CHECK: %[[RHS_NAN:.+]] = spirv.IsNan %[[RHS]] : vector<2xf32>
  // CHECK: %[[SELECT1:.+]] = spirv.Select %[[LHS_NAN]], %[[LHS]], %[[MAX]]
  // CHECK: %[[SELECT2:.+]] = spirv.Select %[[RHS_NAN]], %[[RHS]], %[[SELECT1]]
  %0 = arith.maxf %arg0, %arg1 : vector<2xf32>
  // CHECK: return %[[SELECT2]]
  return %0: vector<2xf32>
}

// Check int vector types.
// CHECK-LABEL: @int_vector234
func.func @int_vector234(%arg0: vector<2xi8>, %arg1: vector<4xi64>) {
  // CHECK: spirv.SDiv %{{.*}}, %{{.*}}: vector<2xi8>
  %0 = arith.divsi %arg0, %arg0: vector<2xi8>
  // CHECK: spirv.UDiv %{{.*}}, %{{.*}}: vector<4xi64>
  %1 = arith.divui %arg1, %arg1: vector<4xi64>
  return
}

// CHECK-LABEL: @vector_srem
// CHECK-SAME: (%[[LHS:.+]]: vector<3xi16>, %[[RHS:.+]]: vector<3xi16>)
func.func @vector_srem(%arg0: vector<3xi16>, %arg1: vector<3xi16>) {
  // CHECK: %[[LABS:.+]] = spirv.GL.SAbs %[[LHS]] : vector<3xi16>
  // CHECK: %[[RABS:.+]] = spirv.GL.SAbs %[[RHS]] : vector<3xi16>
  // CHECK:  %[[ABS:.+]] = spirv.UMod %[[LABS]], %[[RABS]] : vector<3xi16>
  // CHECK:  %[[POS:.+]] = spirv.IEqual %[[LHS]], %[[LABS]] : vector<3xi16>
  // CHECK:  %[[NEG:.+]] = spirv.SNegate %[[ABS]] : vector<3xi16>
  // CHECK:      %{{.+}} = spirv.Select %[[POS]], %[[ABS]], %[[NEG]] : vector<3xi1>, vector<3xi16>
  %0 = arith.remsi %arg0, %arg1: vector<3xi16>
  return
}

// Check float vector types.
// CHECK-LABEL: @float_vector234
func.func @float_vector234(%arg0: vector<2xf16>, %arg1: vector<3xf64>) {
  // CHECK: spirv.FAdd %{{.*}}, %{{.*}}: vector<2xf16>
  %0 = arith.addf %arg0, %arg0: vector<2xf16>
  // CHECK: spirv.FMul %{{.*}}, %{{.*}}: vector<3xf64>
  %1 = arith.mulf %arg1, %arg1: vector<3xf64>
  return
}

// CHECK-LABEL: @one_elem_vector
func.func @one_elem_vector(%arg0: vector<1xi32>) {
  // CHECK: spirv.IAdd %{{.+}}, %{{.+}}: i32
  %0 = arith.addi %arg0, %arg0: vector<1xi32>
  return
}

} // end module

// -----

// Check that types are converted to 32-bit when no special capabilities.
module attributes {
  spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [], []>, #spirv.resource_limits<>>
} {

// CHECK-LABEL: @int_vector23
func.func @int_vector23(%arg0: vector<2xi8>, %arg1: vector<3xi16>) {
  // CHECK: spirv.SDiv %{{.*}}, %{{.*}}: vector<2xi32>
  %0 = arith.divsi %arg0, %arg0: vector<2xi8>
  // CHECK: spirv.SDiv %{{.*}}, %{{.*}}: vector<3xi32>
  %1 = arith.divsi %arg1, %arg1: vector<3xi16>
  return
}

// CHECK-LABEL: @float_scalar
func.func @float_scalar(%arg0: f16) {
  // CHECK: spirv.FAdd %{{.*}}, %{{.*}}: f32
  %0 = arith.addf %arg0, %arg0: f16
  return
}

} // end module
