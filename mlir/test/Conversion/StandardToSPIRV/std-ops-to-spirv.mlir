// RUN: mlir-opt -split-input-file -convert-std-to-spirv -verify-diagnostics %s | FileCheck %s

//===----------------------------------------------------------------------===//
// arith.select
//===----------------------------------------------------------------------===//

module attributes {
  spv.target_env = #spv.target_env<
    #spv.vce<v1.0, [Int8, Int16, Int64, Float16, Float64, Shader], []>, {}>
} {

// Check integer operation conversions.
// CHECK-LABEL: @int32_scalar
func @int32_scalar(%lhs: i32, %rhs: i32) {
  // CHECK: spv.IAdd %{{.*}}, %{{.*}}: i32
  %0 = arith.addi %lhs, %rhs: i32
  // CHECK: spv.ISub %{{.*}}, %{{.*}}: i32
  %1 = arith.subi %lhs, %rhs: i32
  // CHECK: spv.IMul %{{.*}}, %{{.*}}: i32
  %2 = arith.muli %lhs, %rhs: i32
  // CHECK: spv.SDiv %{{.*}}, %{{.*}}: i32
  %3 = arith.divsi %lhs, %rhs: i32
  // CHECK: spv.UDiv %{{.*}}, %{{.*}}: i32
  %4 = arith.divui %lhs, %rhs: i32
  // CHECK: spv.UMod %{{.*}}, %{{.*}}: i32
  %5 = arith.remui %lhs, %rhs: i32
  // CHECK: spv.GLSL.SMax %{{.*}}, %{{.*}}: i32
  %6 = arith.maxsi %lhs, %rhs : i32
  // CHECK: spv.GLSL.UMax %{{.*}}, %{{.*}}: i32
  %7 = arith.maxui %lhs, %rhs : i32
  // CHECK: spv.GLSL.SMin %{{.*}}, %{{.*}}: i32
  %8 = arith.minsi %lhs, %rhs : i32
  // CHECK: spv.GLSL.UMin %{{.*}}, %{{.*}}: i32
  %9 = arith.minui %lhs, %rhs : i32
  return
}

// CHECK-LABEL: @scalar_srem
// CHECK-SAME: (%[[LHS:.+]]: i32, %[[RHS:.+]]: i32)
func @scalar_srem(%lhs: i32, %rhs: i32) {
  // CHECK: %[[LABS:.+]] = spv.GLSL.SAbs %[[LHS]] : i32
  // CHECK: %[[RABS:.+]] = spv.GLSL.SAbs %[[RHS]] : i32
  // CHECK:  %[[ABS:.+]] = spv.UMod %[[LABS]], %[[RABS]] : i32
  // CHECK:  %[[POS:.+]] = spv.IEqual %[[LHS]], %[[LABS]] : i32
  // CHECK:  %[[NEG:.+]] = spv.SNegate %[[ABS]] : i32
  // CHECK:      %{{.+}} = spv.Select %[[POS]], %[[ABS]], %[[NEG]] : i1, i32
  %0 = arith.remsi %lhs, %rhs: i32
  return
}

// Check float unary operation conversions.
// CHECK-LABEL: @float32_unary_scalar
func @float32_unary_scalar(%arg0: f32) {
  // CHECK: spv.GLSL.FAbs %{{.*}}: f32
  %0 = math.abs %arg0 : f32
  // CHECK: spv.GLSL.Ceil %{{.*}}: f32
  %1 = math.ceil %arg0 : f32
  // CHECK: spv.FNegate %{{.*}}: f32
  %5 = arith.negf %arg0 : f32
  // CHECK: spv.GLSL.Floor %{{.*}}: f32
  %10 = math.floor %arg0 : f32
  return
}

// Check float binary operation conversions.
// CHECK-LABEL: @float32_binary_scalar
func @float32_binary_scalar(%lhs: f32, %rhs: f32) {
  // CHECK: spv.FAdd %{{.*}}, %{{.*}}: f32
  %0 = arith.addf %lhs, %rhs: f32
  // CHECK: spv.FSub %{{.*}}, %{{.*}}: f32
  %1 = arith.subf %lhs, %rhs: f32
  // CHECK: spv.FMul %{{.*}}, %{{.*}}: f32
  %2 = arith.mulf %lhs, %rhs: f32
  // CHECK: spv.FDiv %{{.*}}, %{{.*}}: f32
  %3 = arith.divf %lhs, %rhs: f32
  // CHECK: spv.FRem %{{.*}}, %{{.*}}: f32
  %4 = arith.remf %lhs, %rhs: f32
  // CHECK: spv.GLSL.FMax %{{.*}}, %{{.*}}: f32
  %5 = arith.maxf %lhs, %rhs: f32
  // CHECK: spv.GLSL.FMin %{{.*}}, %{{.*}}: f32
  %6 = arith.minf %lhs, %rhs: f32
  return
}

// Check int vector types.
// CHECK-LABEL: @int_vector234
func @int_vector234(%arg0: vector<2xi8>, %arg1: vector<4xi64>) {
  // CHECK: spv.SDiv %{{.*}}, %{{.*}}: vector<2xi8>
  %0 = arith.divsi %arg0, %arg0: vector<2xi8>
  // CHECK: spv.UDiv %{{.*}}, %{{.*}}: vector<4xi64>
  %1 = arith.divui %arg1, %arg1: vector<4xi64>
  return
}

// CHECK-LABEL: @vector_srem
// CHECK-SAME: (%[[LHS:.+]]: vector<3xi16>, %[[RHS:.+]]: vector<3xi16>)
func @vector_srem(%arg0: vector<3xi16>, %arg1: vector<3xi16>) {
  // CHECK: %[[LABS:.+]] = spv.GLSL.SAbs %[[LHS]] : vector<3xi16>
  // CHECK: %[[RABS:.+]] = spv.GLSL.SAbs %[[RHS]] : vector<3xi16>
  // CHECK:  %[[ABS:.+]] = spv.UMod %[[LABS]], %[[RABS]] : vector<3xi16>
  // CHECK:  %[[POS:.+]] = spv.IEqual %[[LHS]], %[[LABS]] : vector<3xi16>
  // CHECK:  %[[NEG:.+]] = spv.SNegate %[[ABS]] : vector<3xi16>
  // CHECK:      %{{.+}} = spv.Select %[[POS]], %[[ABS]], %[[NEG]] : vector<3xi1>, vector<3xi16>
  %0 = arith.remsi %arg0, %arg1: vector<3xi16>
  return
}

// Check float vector types.
// CHECK-LABEL: @float_vector234
func @float_vector234(%arg0: vector<2xf16>, %arg1: vector<3xf64>) {
  // CHECK: spv.FAdd %{{.*}}, %{{.*}}: vector<2xf16>
  %0 = arith.addf %arg0, %arg0: vector<2xf16>
  // CHECK: spv.FMul %{{.*}}, %{{.*}}: vector<3xf64>
  %1 = arith.mulf %arg1, %arg1: vector<3xf64>
  return
}

// CHECK-LABEL: @one_elem_vector
func @one_elem_vector(%arg0: vector<1xi32>) {
  // CHECK: spv.IAdd %{{.+}}, %{{.+}}: i32
  %0 = arith.addi %arg0, %arg0: vector<1xi32>
  return
}

// CHECK-LABEL: @unsupported_5elem_vector
func @unsupported_5elem_vector(%arg0: vector<5xi32>) {
  // CHECK: subi
  %1 = arith.subi %arg0, %arg0: vector<5xi32>
  return
}

// CHECK-LABEL: @unsupported_2x2elem_vector
func @unsupported_2x2elem_vector(%arg0: vector<2x2xi32>) {
  // CHECK: muli
  %2 = arith.muli %arg0, %arg0: vector<2x2xi32>
  return
}

} // end module

// -----

// Check that types are converted to 32-bit when no special capabilities.
module attributes {
  spv.target_env = #spv.target_env<#spv.vce<v1.0, [], []>, {}>
} {

// CHECK-LABEL: @int_vector23
func @int_vector23(%arg0: vector<2xi8>, %arg1: vector<3xi16>) {
  // CHECK: spv.SDiv %{{.*}}, %{{.*}}: vector<2xi32>
  %0 = arith.divsi %arg0, %arg0: vector<2xi8>
  // CHECK: spv.SDiv %{{.*}}, %{{.*}}: vector<3xi32>
  %1 = arith.divsi %arg1, %arg1: vector<3xi16>
  return
}

// CHECK-LABEL: @float_scalar
func @float_scalar(%arg0: f16, %arg1: f64) {
  // CHECK: spv.FAdd %{{.*}}, %{{.*}}: f32
  %0 = arith.addf %arg0, %arg0: f16
  // CHECK: spv.FMul %{{.*}}, %{{.*}}: f32
  %1 = arith.mulf %arg1, %arg1: f64
  return
}

} // end module

// -----

// Check that types are converted to 32-bit when no special capabilities that
// are not supported.
module attributes {
  spv.target_env = #spv.target_env<#spv.vce<v1.0, [], []>, {}>
} {

// expected-error@below {{failed to materialize conversion for block argument #0 that remained live after conversion}}
func @int_vector4_invalid(%arg0: vector<4xi64>) {
  // expected-error@below {{bitwidth emulation is not implemented yet on unsigned op}}
  // expected-note@below {{see existing live user here}}
  %0 = arith.divui %arg0, %arg0: vector<4xi64>
  return
}

} // end module

// -----

//===----------------------------------------------------------------------===//
// std bit ops
//===----------------------------------------------------------------------===//

module attributes {
  spv.target_env = #spv.target_env<#spv.vce<v1.0, [], []>, {}>
} {

// CHECK-LABEL: @bitwise_scalar
func @bitwise_scalar(%arg0 : i32, %arg1 : i32) {
  // CHECK: spv.BitwiseAnd
  %0 = arith.andi %arg0, %arg1 : i32
  // CHECK: spv.BitwiseOr
  %1 = arith.ori %arg0, %arg1 : i32
  // CHECK: spv.BitwiseXor
  %2 = arith.xori %arg0, %arg1 : i32
  return
}

// CHECK-LABEL: @bitwise_vector
func @bitwise_vector(%arg0 : vector<4xi32>, %arg1 : vector<4xi32>) {
  // CHECK: spv.BitwiseAnd
  %0 = arith.andi %arg0, %arg1 : vector<4xi32>
  // CHECK: spv.BitwiseOr
  %1 = arith.ori %arg0, %arg1 : vector<4xi32>
  // CHECK: spv.BitwiseXor
  %2 = arith.xori %arg0, %arg1 : vector<4xi32>
  return
}

// CHECK-LABEL: @logical_scalar
func @logical_scalar(%arg0 : i1, %arg1 : i1) {
  // CHECK: spv.LogicalAnd
  %0 = arith.andi %arg0, %arg1 : i1
  // CHECK: spv.LogicalOr
  %1 = arith.ori %arg0, %arg1 : i1
  // CHECK: spv.LogicalNotEqual
  %2 = arith.xori %arg0, %arg1 : i1
  return
}

// CHECK-LABEL: @logical_vector
func @logical_vector(%arg0 : vector<4xi1>, %arg1 : vector<4xi1>) {
  // CHECK: spv.LogicalAnd
  %0 = arith.andi %arg0, %arg1 : vector<4xi1>
  // CHECK: spv.LogicalOr
  %1 = arith.ori %arg0, %arg1 : vector<4xi1>
  // CHECK: spv.LogicalNotEqual
  %2 = arith.xori %arg0, %arg1 : vector<4xi1>
  return
}

// CHECK-LABEL: @shift_scalar
func @shift_scalar(%arg0 : i32, %arg1 : i32) {
  // CHECK: spv.ShiftLeftLogical
  %0 = arith.shli %arg0, %arg1 : i32
  // CHECK: spv.ShiftRightArithmetic
  %1 = arith.shrsi %arg0, %arg1 : i32
  // CHECK: spv.ShiftRightLogical
  %2 = arith.shrui %arg0, %arg1 : i32
  return
}

// CHECK-LABEL: @shift_vector
func @shift_vector(%arg0 : vector<4xi32>, %arg1 : vector<4xi32>) {
  // CHECK: spv.ShiftLeftLogical
  %0 = arith.shli %arg0, %arg1 : vector<4xi32>
  // CHECK: spv.ShiftRightArithmetic
  %1 = arith.shrsi %arg0, %arg1 : vector<4xi32>
  // CHECK: spv.ShiftRightLogical
  %2 = arith.shrui %arg0, %arg1 : vector<4xi32>
  return
}

} // end module

// -----

//===----------------------------------------------------------------------===//
// std.cmpf
//===----------------------------------------------------------------------===//

module attributes {
  spv.target_env = #spv.target_env<#spv.vce<v1.0, [], []>, {}>
} {

// CHECK-LABEL: @cmpf
func @cmpf(%arg0 : f32, %arg1 : f32) {
  // CHECK: spv.FOrdEqual
  %1 = arith.cmpf oeq, %arg0, %arg1 : f32
  // CHECK: spv.FOrdGreaterThan
  %2 = arith.cmpf ogt, %arg0, %arg1 : f32
  // CHECK: spv.FOrdGreaterThanEqual
  %3 = arith.cmpf oge, %arg0, %arg1 : f32
  // CHECK: spv.FOrdLessThan
  %4 = arith.cmpf olt, %arg0, %arg1 : f32
  // CHECK: spv.FOrdLessThanEqual
  %5 = arith.cmpf ole, %arg0, %arg1 : f32
  // CHECK: spv.FOrdNotEqual
  %6 = arith.cmpf one, %arg0, %arg1 : f32
  // CHECK: spv.FUnordEqual
  %7 = arith.cmpf ueq, %arg0, %arg1 : f32
  // CHECK: spv.FUnordGreaterThan
  %8 = arith.cmpf ugt, %arg0, %arg1 : f32
  // CHECK: spv.FUnordGreaterThanEqual
  %9 = arith.cmpf uge, %arg0, %arg1 : f32
  // CHECK: spv.FUnordLessThan
  %10 = arith.cmpf ult, %arg0, %arg1 : f32
  // CHECK: FUnordLessThanEqual
  %11 = arith.cmpf ule, %arg0, %arg1 : f32
  // CHECK: spv.FUnordNotEqual
  %12 = arith.cmpf une, %arg0, %arg1 : f32
  return
}

} // end module

// -----

// With Kernel capability, we can convert NaN check to spv.Ordered/spv.Unordered.
module attributes {
  spv.target_env = #spv.target_env<#spv.vce<v1.0, [Kernel], []>, {}>
} {

// CHECK-LABEL: @cmpf
func @cmpf(%arg0 : f32, %arg1 : f32) {
  // CHECK: spv.Ordered
  %0 = arith.cmpf ord, %arg0, %arg1 : f32
  // CHECK: spv.Unordered
  %1 = arith.cmpf uno, %arg0, %arg1 : f32
  return
}

} // end module

// -----

// Without Kernel capability, we need to convert NaN check to spv.IsNan.
module attributes {
  spv.target_env = #spv.target_env<#spv.vce<v1.0, [], []>, {}>
} {

// CHECK-LABEL: @cmpf
// CHECK-SAME: %[[LHS:.+]]: f32, %[[RHS:.+]]: f32
func @cmpf(%arg0 : f32, %arg1 : f32) {
  // CHECK:      %[[LHS_NAN:.+]] = spv.IsNan %[[LHS]] : f32
  // CHECK-NEXT: %[[RHS_NAN:.+]] = spv.IsNan %[[RHS]] : f32
  // CHECK-NEXT: %[[OR:.+]] = spv.LogicalOr %[[LHS_NAN]], %[[RHS_NAN]] : i1
  // CHECK-NEXT: %{{.+}} = spv.LogicalNot %[[OR]] : i1
  %0 = arith.cmpf ord, %arg0, %arg1 : f32

  // CHECK-NEXT: %[[LHS_NAN:.+]] = spv.IsNan %[[LHS]] : f32
  // CHECK-NEXT: %[[RHS_NAN:.+]] = spv.IsNan %[[RHS]] : f32
  // CHECK-NEXT: %{{.+}} = spv.LogicalOr %[[LHS_NAN]], %[[RHS_NAN]] : i1
  %1 = arith.cmpf uno, %arg0, %arg1 : f32
  return
}

} // end module

// -----

//===----------------------------------------------------------------------===//
// std.cmpi
//===----------------------------------------------------------------------===//

module attributes {
  spv.target_env = #spv.target_env<#spv.vce<v1.0, [], []>, {}>
} {

// CHECK-LABEL: @cmpi
func @cmpi(%arg0 : i32, %arg1 : i32) {
  // CHECK: spv.IEqual
  %0 = arith.cmpi eq, %arg0, %arg1 : i32
  // CHECK: spv.INotEqual
  %1 = arith.cmpi ne, %arg0, %arg1 : i32
  // CHECK: spv.SLessThan
  %2 = arith.cmpi slt, %arg0, %arg1 : i32
  // CHECK: spv.SLessThanEqual
  %3 = arith.cmpi sle, %arg0, %arg1 : i32
  // CHECK: spv.SGreaterThan
  %4 = arith.cmpi sgt, %arg0, %arg1 : i32
  // CHECK: spv.SGreaterThanEqual
  %5 = arith.cmpi sge, %arg0, %arg1 : i32
  // CHECK: spv.ULessThan
  %6 = arith.cmpi ult, %arg0, %arg1 : i32
  // CHECK: spv.ULessThanEqual
  %7 = arith.cmpi ule, %arg0, %arg1 : i32
  // CHECK: spv.UGreaterThan
  %8 = arith.cmpi ugt, %arg0, %arg1 : i32
  // CHECK: spv.UGreaterThanEqual
  %9 = arith.cmpi uge, %arg0, %arg1 : i32
  return
}

// CHECK-LABEL: @boolcmpi
func @boolcmpi(%arg0 : i1, %arg1 : i1) {
  // CHECK: spv.LogicalEqual
  %0 = arith.cmpi eq, %arg0, %arg1 : i1
  // CHECK: spv.LogicalNotEqual
  %1 = arith.cmpi ne, %arg0, %arg1 : i1
  return
}

// CHECK-LABEL: @vecboolcmpi
func @vecboolcmpi(%arg0 : vector<4xi1>, %arg1 : vector<4xi1>) {
  // CHECK: spv.LogicalEqual
  %0 = arith.cmpi eq, %arg0, %arg1 : vector<4xi1>
  // CHECK: spv.LogicalNotEqual
  %1 = arith.cmpi ne, %arg0, %arg1 : vector<4xi1>
  return
}

} // end module

// -----

//===----------------------------------------------------------------------===//
// arith.constant
//===----------------------------------------------------------------------===//

module attributes {
  spv.target_env = #spv.target_env<
    #spv.vce<v1.0, [Int8, Int16, Int64, Float16, Float64], []>, {}>
} {

// CHECK-LABEL: @constant
func @constant() {
  // CHECK: spv.Constant true
  %0 = arith.constant true
  // CHECK: spv.Constant 42 : i32
  %1 = arith.constant 42 : i32
  // CHECK: spv.Constant 5.000000e-01 : f32
  %2 = arith.constant 0.5 : f32
  // CHECK: spv.Constant dense<[2, 3]> : vector<2xi32>
  %3 = arith.constant dense<[2, 3]> : vector<2xi32>
  // CHECK: spv.Constant 1 : i32
  %4 = arith.constant 1 : index
  // CHECK: spv.Constant dense<1> : tensor<6xi32> : !spv.array<6 x i32, stride=4>
  %5 = arith.constant dense<1> : tensor<2x3xi32>
  // CHECK: spv.Constant dense<1.000000e+00> : tensor<6xf32> : !spv.array<6 x f32, stride=4>
  %6 = arith.constant dense<1.0> : tensor<2x3xf32>
  // CHECK: spv.Constant dense<{{\[}}1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00]> : tensor<6xf32> : !spv.array<6 x f32, stride=4>
  %7 = arith.constant dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf32>
  // CHECK: spv.Constant dense<{{\[}}1, 2, 3, 4, 5, 6]> : tensor<6xi32> : !spv.array<6 x i32, stride=4>
  %8 = arith.constant dense<[[1, 2, 3], [4, 5, 6]]> : tensor<2x3xi32>
  // CHECK: spv.Constant dense<{{\[}}1, 2, 3, 4, 5, 6]> : tensor<6xi32> : !spv.array<6 x i32, stride=4>
  %9 = arith.constant dense<[[1, 2], [3, 4], [5, 6]]> : tensor<3x2xi32>
  // CHECK: spv.Constant dense<{{\[}}1, 2, 3, 4, 5, 6]> : tensor<6xi32> : !spv.array<6 x i32, stride=4>
  %10 = arith.constant dense<[1, 2, 3, 4, 5, 6]> : tensor<6xi32>
  return
}

// CHECK-LABEL: @constant_16bit
func @constant_16bit() {
  // CHECK: spv.Constant 4 : i16
  %0 = arith.constant 4 : i16
  // CHECK: spv.Constant 5.000000e+00 : f16
  %1 = arith.constant 5.0 : f16
  // CHECK: spv.Constant dense<[2, 3]> : vector<2xi16>
  %2 = arith.constant dense<[2, 3]> : vector<2xi16>
  // CHECK: spv.Constant dense<4.000000e+00> : tensor<5xf16> : !spv.array<5 x f16, stride=2>
  %3 = arith.constant dense<4.0> : tensor<5xf16>
  return
}

// CHECK-LABEL: @constant_64bit
func @constant_64bit() {
  // CHECK: spv.Constant 4 : i64
  %0 = arith.constant 4 : i64
  // CHECK: spv.Constant 5.000000e+00 : f64
  %1 = arith.constant 5.0 : f64
  // CHECK: spv.Constant dense<[2, 3]> : vector<2xi64>
  %2 = arith.constant dense<[2, 3]> : vector<2xi64>
  // CHECK: spv.Constant dense<4.000000e+00> : tensor<5xf64> : !spv.array<5 x f64, stride=8>
  %3 = arith.constant dense<4.0> : tensor<5xf64>
  return
}

} // end module

// -----

// Check that constants are converted to 32-bit when no special capability.
module attributes {
  spv.target_env = #spv.target_env<#spv.vce<v1.0, [], []>, {}>
} {

// CHECK-LABEL: @constant_16bit
func @constant_16bit() {
  // CHECK: spv.Constant 4 : i32
  %0 = arith.constant 4 : i16
  // CHECK: spv.Constant 5.000000e+00 : f32
  %1 = arith.constant 5.0 : f16
  // CHECK: spv.Constant dense<[2, 3]> : vector<2xi32>
  %2 = arith.constant dense<[2, 3]> : vector<2xi16>
  // CHECK: spv.Constant dense<4.000000e+00> : tensor<5xf32> : !spv.array<5 x f32, stride=4>
  %3 = arith.constant dense<4.0> : tensor<5xf16>
  // CHECK: spv.Constant dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00]> : tensor<4xf32> : !spv.array<4 x f32, stride=4>
  %4 = arith.constant dense<[[1.0, 2.0], [3.0, 4.0]]> : tensor<2x2xf16>
  return
}

// CHECK-LABEL: @constant_64bit
func @constant_64bit() {
  // CHECK: spv.Constant 4 : i32
  %0 = arith.constant 4 : i64
  // CHECK: spv.Constant 5.000000e+00 : f32
  %1 = arith.constant 5.0 : f64
  // CHECK: spv.Constant dense<[2, 3]> : vector<2xi32>
  %2 = arith.constant dense<[2, 3]> : vector<2xi64>
  // CHECK: spv.Constant dense<4.000000e+00> : tensor<5xf32> : !spv.array<5 x f32, stride=4>
  %3 = arith.constant dense<4.0> : tensor<5xf64>
  // CHECK: spv.Constant dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00]> : tensor<4xf32> : !spv.array<4 x f32, stride=4>
  %4 = arith.constant dense<[[1.0, 2.0], [3.0, 4.0]]> : tensor<2x2xf16>
  return
}

// CHECK-LABEL: @corner_cases
func @corner_cases() {
  // CHECK: %{{.*}} = spv.Constant -1 : i32
  %0 = arith.constant 4294967295  : i64 // 2^32 - 1
  // CHECK: %{{.*}} = spv.Constant 2147483647 : i32
  %1 = arith.constant 2147483647  : i64 // 2^31 - 1
  // CHECK: %{{.*}} = spv.Constant -2147483648 : i32
  %2 = arith.constant 2147483648  : i64 // 2^31
  // CHECK: %{{.*}} = spv.Constant -2147483648 : i32
  %3 = arith.constant -2147483648 : i64 // -2^31

  // CHECK: %{{.*}} = spv.Constant -1 : i32
  %5 = arith.constant -1 : i64
  // CHECK: %{{.*}} = spv.Constant -2 : i32
  %6 = arith.constant -2 : i64
  // CHECK: %{{.*}} = spv.Constant -1 : i32
  %7 = arith.constant -1 : index
  // CHECK: %{{.*}} = spv.Constant -2 : i32
  %8 = arith.constant -2 : index


  // CHECK: spv.Constant false
  %9 = arith.constant false
  // CHECK: spv.Constant true
  %10 = arith.constant true

  return
}

// CHECK-LABEL: @unsupported_cases
func @unsupported_cases() {
  // CHECK: %{{.*}} = arith.constant 4294967296 : i64
  %0 = arith.constant 4294967296 : i64 // 2^32
  // CHECK: %{{.*}} = arith.constant -2147483649 : i64
  %1 = arith.constant -2147483649 : i64 // -2^31 - 1
  // CHECK: %{{.*}} = arith.constant 1.0000000000000002 : f64
  %2 = arith.constant 0x3FF0000000000001 : f64 // smallest number > 1
  return
}

} // end module

// -----

//===----------------------------------------------------------------------===//
// std cast ops
//===----------------------------------------------------------------------===//

module attributes {
  spv.target_env = #spv.target_env<
    #spv.vce<v1.0, [Int8, Int16, Int64, Float16, Float64], []>, {}>
} {

// CHECK-LABEL: index_cast1
func @index_cast1(%arg0: i16) {
  // CHECK: spv.SConvert %{{.+}} : i16 to i32
  %0 = arith.index_cast %arg0 : i16 to index
  return
}

// CHECK-LABEL: index_cast2
func @index_cast2(%arg0: index) {
  // CHECK: spv.SConvert %{{.+}} : i32 to i16
  %0 = arith.index_cast %arg0 : index to i16
  return
}

// CHECK-LABEL: index_cast3
func @index_cast3(%arg0: i32) {
  // CHECK-NOT: spv.SConvert
  %0 = arith.index_cast %arg0 : i32 to index
  return
}

// CHECK-LABEL: index_cast4
func @index_cast4(%arg0: index) {
  // CHECK-NOT: spv.SConvert
  %0 = arith.index_cast %arg0 : index to i32
  return
}

// CHECK-LABEL: @fpext1
func @fpext1(%arg0: f16) -> f64 {
  // CHECK: spv.FConvert %{{.*}} : f16 to f64
  %0 = arith.extf %arg0 : f16 to f64
  return %0 : f64
}

// CHECK-LABEL: @fpext2
func @fpext2(%arg0 : f32) -> f64 {
  // CHECK: spv.FConvert %{{.*}} : f32 to f64
  %0 = arith.extf %arg0 : f32 to f64
  return %0 : f64
}

// CHECK-LABEL: @fptrunc1
func @fptrunc1(%arg0 : f64) -> f16 {
  // CHECK: spv.FConvert %{{.*}} : f64 to f16
  %0 = arith.truncf %arg0 : f64 to f16
  return %0 : f16
}

// CHECK-LABEL: @fptrunc2
func @fptrunc2(%arg0: f32) -> f16 {
  // CHECK: spv.FConvert %{{.*}} : f32 to f16
  %0 = arith.truncf %arg0 : f32 to f16
  return %0 : f16
}

// CHECK-LABEL: @sitofp1
func @sitofp1(%arg0 : i32) -> f32 {
  // CHECK: spv.ConvertSToF %{{.*}} : i32 to f32
  %0 = arith.sitofp %arg0 : i32 to f32
  return %0 : f32
}

// CHECK-LABEL: @sitofp2
func @sitofp2(%arg0 : i64) -> f64 {
  // CHECK: spv.ConvertSToF %{{.*}} : i64 to f64
  %0 = arith.sitofp %arg0 : i64 to f64
  return %0 : f64
}

// CHECK-LABEL: @uitofp_i16_f32
func @uitofp_i16_f32(%arg0: i16) -> f32 {
  // CHECK: spv.ConvertUToF %{{.*}} : i16 to f32
  %0 = arith.uitofp %arg0 : i16 to f32
  return %0 : f32
}

// CHECK-LABEL: @uitofp_i32_f32
func @uitofp_i32_f32(%arg0 : i32) -> f32 {
  // CHECK: spv.ConvertUToF %{{.*}} : i32 to f32
  %0 = arith.uitofp %arg0 : i32 to f32
  return %0 : f32
}

// CHECK-LABEL: @uitofp_i1_f32
func @uitofp_i1_f32(%arg0 : i1) -> f32 {
  // CHECK: %[[ZERO:.+]] = spv.Constant 0.000000e+00 : f32
  // CHECK: %[[ONE:.+]] = spv.Constant 1.000000e+00 : f32
  // CHECK: spv.Select %{{.*}}, %[[ONE]], %[[ZERO]] : i1, f32
  %0 = arith.uitofp %arg0 : i1 to f32
  return %0 : f32
}

// CHECK-LABEL: @uitofp_i1_f64
func @uitofp_i1_f64(%arg0 : i1) -> f64 {
  // CHECK: %[[ZERO:.+]] = spv.Constant 0.000000e+00 : f64
  // CHECK: %[[ONE:.+]] = spv.Constant 1.000000e+00 : f64
  // CHECK: spv.Select %{{.*}}, %[[ONE]], %[[ZERO]] : i1, f64
  %0 = arith.uitofp %arg0 : i1 to f64
  return %0 : f64
}

// CHECK-LABEL: @uitofp_vec_i1_f32
func @uitofp_vec_i1_f32(%arg0 : vector<4xi1>) -> vector<4xf32> {
  // CHECK: %[[ZERO:.+]] = spv.Constant dense<0.000000e+00> : vector<4xf32>
  // CHECK: %[[ONE:.+]] = spv.Constant dense<1.000000e+00> : vector<4xf32>
  // CHECK: spv.Select %{{.*}}, %[[ONE]], %[[ZERO]] : vector<4xi1>, vector<4xf32>
  %0 = arith.uitofp %arg0 : vector<4xi1> to vector<4xf32>
  return %0 : vector<4xf32>
}

// CHECK-LABEL: @uitofp_vec_i1_f64
spv.func @uitofp_vec_i1_f64(%arg0: vector<4xi1>) -> vector<4xf64> "None" {
  // CHECK: %[[ZERO:.+]] = spv.Constant dense<0.000000e+00> : vector<4xf64>
  // CHECK: %[[ONE:.+]] = spv.Constant dense<1.000000e+00> : vector<4xf64>
  // CHECK: spv.Select %{{.*}}, %[[ONE]], %[[ZERO]] : vector<4xi1>, vector<4xf64>
  %0 = spv.Constant dense<0.000000e+00> : vector<4xf64>
  %1 = spv.Constant dense<1.000000e+00> : vector<4xf64>
  %2 = spv.Select %arg0, %1, %0 : vector<4xi1>, vector<4xf64>
  spv.ReturnValue %2 : vector<4xf64>
}

// CHECK-LABEL: @sexti1
func @sexti1(%arg0: i16) -> i64 {
  // CHECK: spv.SConvert %{{.*}} : i16 to i64
  %0 = arith.extsi %arg0 : i16 to i64
  return %0 : i64
}

// CHECK-LABEL: @sexti2
func @sexti2(%arg0 : i32) -> i64 {
  // CHECK: spv.SConvert %{{.*}} : i32 to i64
  %0 = arith.extsi %arg0 : i32 to i64
  return %0 : i64
}

// CHECK-LABEL: @zexti1
func @zexti1(%arg0: i16) -> i64 {
  // CHECK: spv.UConvert %{{.*}} : i16 to i64
  %0 = arith.extui %arg0 : i16 to i64
  return %0 : i64
}

// CHECK-LABEL: @zexti2
func @zexti2(%arg0 : i32) -> i64 {
  // CHECK: spv.UConvert %{{.*}} : i32 to i64
  %0 = arith.extui %arg0 : i32 to i64
  return %0 : i64
}

// CHECK-LABEL: @zexti3
func @zexti3(%arg0 : i1) -> i32 {
  // CHECK: %[[ZERO:.+]] = spv.Constant 0 : i32
  // CHECK: %[[ONE:.+]] = spv.Constant 1 : i32
  // CHECK: spv.Select %{{.*}}, %[[ONE]], %[[ZERO]] : i1, i32
  %0 = arith.extui %arg0 : i1 to i32
  return %0 : i32
}

// CHECK-LABEL: @zexti4
func @zexti4(%arg0 : vector<4xi1>) -> vector<4xi32> {
  // CHECK: %[[ZERO:.+]] = spv.Constant dense<0> : vector<4xi32>
  // CHECK: %[[ONE:.+]] = spv.Constant dense<1> : vector<4xi32>
  // CHECK: spv.Select %{{.*}}, %[[ONE]], %[[ZERO]] : vector<4xi1>, vector<4xi32>
  %0 = arith.extui %arg0 : vector<4xi1> to vector<4xi32>
  return %0 : vector<4xi32>
}

// CHECK-LABEL: @zexti5
func @zexti5(%arg0 : vector<4xi1>) -> vector<4xi64> {
  // CHECK: %[[ZERO:.+]] = spv.Constant dense<0> : vector<4xi64>
  // CHECK: %[[ONE:.+]] = spv.Constant dense<1> : vector<4xi64>
  // CHECK: spv.Select %{{.*}}, %[[ONE]], %[[ZERO]] : vector<4xi1>, vector<4xi64>
  %0 = arith.extui %arg0 : vector<4xi1> to vector<4xi64>
  return %0 : vector<4xi64>
}

// CHECK-LABEL: @trunci1
func @trunci1(%arg0 : i64) -> i16 {
  // CHECK: spv.SConvert %{{.*}} : i64 to i16
  %0 = arith.trunci %arg0 : i64 to i16
  return %0 : i16
}

// CHECK-LABEL: @trunci2
func @trunci2(%arg0: i32) -> i16 {
  // CHECK: spv.SConvert %{{.*}} : i32 to i16
  %0 = arith.trunci %arg0 : i32 to i16
  return %0 : i16
}

// CHECK-LABEL: @trunc_to_i1
func @trunc_to_i1(%arg0: i32) -> i1 {
  // CHECK: %[[MASK:.*]] = spv.Constant 1 : i32
  // CHECK: %[[MASKED_SRC:.*]] = spv.BitwiseAnd %{{.*}}, %[[MASK]] : i32
  // CHECK: %[[IS_ONE:.*]] = spv.IEqual %[[MASKED_SRC]], %[[MASK]] : i32
  // CHECK-DAG: %[[TRUE:.*]] = spv.Constant true
  // CHECK-DAG: %[[FALSE:.*]] = spv.Constant false
  // CHECK: spv.Select %[[IS_ONE]], %[[TRUE]], %[[FALSE]] : i1, i1
  %0 = arith.trunci %arg0 : i32 to i1
  return %0 : i1
}

// CHECK-LABEL: @trunc_to_veci1
func @trunc_to_veci1(%arg0: vector<4xi32>) -> vector<4xi1> {
  // CHECK: %[[MASK:.*]] = spv.Constant dense<1> : vector<4xi32>
  // CHECK: %[[MASKED_SRC:.*]] = spv.BitwiseAnd %{{.*}}, %[[MASK]] : vector<4xi32>
  // CHECK: %[[IS_ONE:.*]] = spv.IEqual %[[MASKED_SRC]], %[[MASK]] : vector<4xi32>
  // CHECK-DAG: %[[TRUE:.*]] = spv.Constant dense<true> : vector<4xi1>
  // CHECK-DAG: %[[FALSE:.*]] = spv.Constant dense<false> : vector<4xi1>
  // CHECK: spv.Select %[[IS_ONE]], %[[TRUE]], %[[FALSE]] : vector<4xi1>, vector<4xi1>
  %0 = arith.trunci %arg0 : vector<4xi32> to vector<4xi1>
  return %0 : vector<4xi1>
}

// CHECK-LABEL: @fptosi1
func @fptosi1(%arg0 : f32) -> i32 {
  // CHECK: spv.ConvertFToS %{{.*}} : f32 to i32
  %0 = arith.fptosi %arg0 : f32 to i32
  return %0 : i32
}

// CHECK-LABEL: @fptosi2
func @fptosi2(%arg0 : f16) -> i16 {
  // CHECK: spv.ConvertFToS %{{.*}} : f16 to i16
  %0 = arith.fptosi %arg0 : f16 to i16
  return %0 : i16
}

} // end module

// -----

// Checks that cast types will be adjusted when missing special capabilities for
// certain non-32-bit scalar types.
module attributes {
  spv.target_env = #spv.target_env<#spv.vce<v1.0, [Float64], []>, {}>
} {

// CHECK-LABEL: @fpext1
// CHECK-SAME: %[[ARG:.*]]: f32
func @fpext1(%arg0: f16) -> f64 {
  // CHECK-NEXT: spv.FConvert %[[ARG]] : f32 to f64
  %0 = arith.extf %arg0 : f16 to f64
  return %0: f64
}

// CHECK-LABEL: @fpext2
// CHECK-SAME: %[[ARG:.*]]: f32
func @fpext2(%arg0 : f32) -> f64 {
  // CHECK-NEXT: spv.FConvert %[[ARG]] : f32 to f64
  %0 = arith.extf %arg0 : f32 to f64
  return %0: f64
}

} // end module

// -----

// Checks that cast types will be adjusted when missing special capabilities for
// certain non-32-bit scalar types.
module attributes {
  spv.target_env = #spv.target_env<#spv.vce<v1.0, [Float16], []>, {}>
} {

// CHECK-LABEL: @fptrunc1
// CHECK-SAME: %[[ARG:.*]]: f32
func @fptrunc1(%arg0 : f64) -> f16 {
  // CHECK-NEXT: spv.FConvert %[[ARG]] : f32 to f16
  %0 = arith.truncf %arg0 : f64 to f16
  return %0: f16
}

// CHECK-LABEL: @fptrunc2
// CHECK-SAME: %[[ARG:.*]]: f32
func @fptrunc2(%arg0: f32) -> f16 {
  // CHECK-NEXT: spv.FConvert %[[ARG]] : f32 to f16
  %0 = arith.truncf %arg0 : f32 to f16
  return %0: f16
}

// CHECK-LABEL: @sitofp
func @sitofp(%arg0 : i64) -> f64 {
  // CHECK: spv.ConvertSToF %{{.*}} : i32 to f32
  %0 = arith.sitofp %arg0 : i64 to f64
  return %0: f64
}

} // end module

// -----

//===----------------------------------------------------------------------===//
// std.return
//===----------------------------------------------------------------------===//

module attributes {
  spv.target_env = #spv.target_env<#spv.vce<v1.0, [], []>, {}>
} {

// CHECK-LABEL: spv.func @return_one_val
//  CHECK-SAME: (%[[ARG:.+]]: f32)
func @return_one_val(%arg0: f32) -> f32 {
  // CHECK: spv.ReturnValue %[[ARG]] : f32
  return %arg0: f32
}

// Check that multiple-return functions are not converted.
// CHECK-LABEL: func @return_multi_val
func @return_multi_val(%arg0: f32) -> (f32, f32) {
  // CHECK: return
  return %arg0, %arg0: f32, f32
}

}

// -----

//===----------------------------------------------------------------------===//
// tensor.extract
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func @tensor_extract_constant
// CHECK-SAME: (%[[A:.+]]: i32, %[[B:.+]]: i32, %[[C:.+]]: i32)
func @tensor_extract_constant(%a : index, %b: index, %c: index) -> i32 {
  // CHECK: %[[CST:.+]] = spv.Constant dense<[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]>
  %cst = arith.constant dense<[[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]> : tensor<2x2x3xi32>
  // CHECK: %[[VAR:.+]] = spv.Variable init(%[[CST]]) : !spv.ptr<!spv.array<12 x i32, stride=4>, Function>
  // CHECK: %[[C0:.+]] = spv.Constant 0 : i32
  // CHECK: %[[C6:.+]] = spv.Constant 6 : i32
  // CHECK: %[[MUL0:.+]] = spv.IMul %[[C6]], %[[A]] : i32
  // CHECK: %[[ADD0:.+]] = spv.IAdd %[[C0]], %[[MUL0]] : i32
  // CHECK: %[[C3:.+]] = spv.Constant 3 : i32
  // CHECK: %[[MUL1:.+]] = spv.IMul %[[C3]], %[[B]] : i32
  // CHECK: %[[ADD1:.+]] = spv.IAdd %[[ADD0]], %[[MUL1]] : i32
  // CHECK: %[[C1:.+]] = spv.Constant 1 : i32
  // CHECK: %[[MUL2:.+]] = spv.IMul %[[C1]], %[[C]] : i32
  // CHECK: %[[ADD2:.+]] = spv.IAdd %[[ADD1]], %[[MUL2]] : i32
  // CHECK: %[[AC:.+]] = spv.AccessChain %[[VAR]][%[[ADD2]]]
  // CHECK: %[[VAL:.+]] = spv.Load "Function" %[[AC]] : i32
  %extract = tensor.extract %cst[%a, %b, %c] : tensor<2x2x3xi32>
  // CHECK: spv.ReturnValue %[[VAL]]
  return %extract : i32
}
