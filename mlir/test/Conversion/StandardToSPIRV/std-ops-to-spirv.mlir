// RUN: mlir-opt -split-input-file -convert-std-to-spirv -verify-diagnostics %s -o - | FileCheck %s

//===----------------------------------------------------------------------===//
// std arithmetic ops
//===----------------------------------------------------------------------===//

module attributes {
  spv.target_env = #spv.target_env<
    #spv.vce<v1.0, [Int8, Int16, Int64, Float16, Float64], []>, {}>
} {

// Check integer operation conversions.
// CHECK-LABEL: @int32_scalar
func @int32_scalar(%lhs: i32, %rhs: i32) {
  // CHECK: spv.IAdd %{{.*}}, %{{.*}}: i32
  %0 = addi %lhs, %rhs: i32
  // CHECK: spv.ISub %{{.*}}, %{{.*}}: i32
  %1 = subi %lhs, %rhs: i32
  // CHECK: spv.IMul %{{.*}}, %{{.*}}: i32
  %2 = muli %lhs, %rhs: i32
  // CHECK: spv.SDiv %{{.*}}, %{{.*}}: i32
  %3 = divi_signed %lhs, %rhs: i32
  // CHECK: spv.UDiv %{{.*}}, %{{.*}}: i32
  %4 = divi_unsigned %lhs, %rhs: i32
  // CHECK: spv.UMod %{{.*}}, %{{.*}}: i32
  %5 = remi_unsigned %lhs, %rhs: i32
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
  %0 = remi_signed %lhs, %rhs: i32
  return
}

// Check float unary operation conversions.
// CHECK-LABEL: @float32_unary_scalar
func @float32_unary_scalar(%arg0: f32) {
  // CHECK: spv.GLSL.FAbs %{{.*}}: f32
  %0 = absf %arg0 : f32
  // CHECK: spv.GLSL.Ceil %{{.*}}: f32
  %1 = ceilf %arg0 : f32
  // CHECK: spv.GLSL.Cos %{{.*}}: f32
  %2 = math.cos %arg0 : f32
  // CHECK: spv.GLSL.Exp %{{.*}}: f32
  %3 = math.exp %arg0 : f32
  // CHECK: spv.GLSL.Log %{{.*}}: f32
  %4 = math.log %arg0 : f32
  // CHECK: spv.FNegate %{{.*}}: f32
  %5 = negf %arg0 : f32
  // CHECK: spv.GLSL.InverseSqrt %{{.*}}: f32
  %6 = math.rsqrt %arg0 : f32
  // CHECK: spv.GLSL.Sqrt %{{.*}}: f32
  %7 = math.sqrt %arg0 : f32
  // CHECK: spv.GLSL.Tanh %{{.*}}: f32
  %8 = math.tanh %arg0 : f32
  // CHECK: spv.GLSL.Sin %{{.*}}: f32
  %9 = math.sin %arg0 : f32
  // CHECK: spv.GLSL.Floor %{{.*}}: f32
  %10 = floorf %arg0 : f32
  return
}

// Check float binary operation conversions.
// CHECK-LABEL: @float32_binary_scalar
func @float32_binary_scalar(%lhs: f32, %rhs: f32) {
  // CHECK: spv.FAdd %{{.*}}, %{{.*}}: f32
  %0 = addf %lhs, %rhs: f32
  // CHECK: spv.FSub %{{.*}}, %{{.*}}: f32
  %1 = subf %lhs, %rhs: f32
  // CHECK: spv.FMul %{{.*}}, %{{.*}}: f32
  %2 = mulf %lhs, %rhs: f32
  // CHECK: spv.FDiv %{{.*}}, %{{.*}}: f32
  %3 = divf %lhs, %rhs: f32
  // CHECK: spv.FRem %{{.*}}, %{{.*}}: f32
  %4 = remf %lhs, %rhs: f32
  return
}

// Check int vector types.
// CHECK-LABEL: @int_vector234
func @int_vector234(%arg0: vector<2xi8>, %arg1: vector<4xi64>) {
  // CHECK: spv.SDiv %{{.*}}, %{{.*}}: vector<2xi8>
  %0 = divi_signed %arg0, %arg0: vector<2xi8>
  // CHECK: spv.UDiv %{{.*}}, %{{.*}}: vector<4xi64>
  %1 = divi_unsigned %arg1, %arg1: vector<4xi64>
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
  %0 = remi_signed %arg0, %arg1: vector<3xi16>
  return
}

// Check float vector types.
// CHECK-LABEL: @float_vector234
func @float_vector234(%arg0: vector<2xf16>, %arg1: vector<3xf64>) {
  // CHECK: spv.FAdd %{{.*}}, %{{.*}}: vector<2xf16>
  %0 = addf %arg0, %arg0: vector<2xf16>
  // CHECK: spv.FMul %{{.*}}, %{{.*}}: vector<3xf64>
  %1 = mulf %arg1, %arg1: vector<3xf64>
  return
}

// CHECK-LABEL: @one_elem_vector
func @one_elem_vector(%arg0: vector<1xi32>) {
  // CHECK: spv.IAdd %{{.+}}, %{{.+}}: i32
  %0 = addi %arg0, %arg0: vector<1xi32>
  return
}

// CHECK-LABEL: @unsupported_5elem_vector
func @unsupported_5elem_vector(%arg0: vector<5xi32>) {
  // CHECK: subi
  %1 = subi %arg0, %arg0: vector<5xi32>
  return
}

// CHECK-LABEL: @unsupported_2x2elem_vector
func @unsupported_2x2elem_vector(%arg0: vector<2x2xi32>) {
  // CHECK: muli
  %2 = muli %arg0, %arg0: vector<2x2xi32>
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
  %0 = divi_signed %arg0, %arg0: vector<2xi8>
  // CHECK: spv.SDiv %{{.*}}, %{{.*}}: vector<3xi32>
  %1 = divi_signed %arg1, %arg1: vector<3xi16>
  return
}

// CHECK-LABEL: @float_scalar
func @float_scalar(%arg0: f16, %arg1: f64) {
  // CHECK: spv.FAdd %{{.*}}, %{{.*}}: f32
  %0 = addf %arg0, %arg0: f16
  // CHECK: spv.FMul %{{.*}}, %{{.*}}: f32
  %1 = mulf %arg1, %arg1: f64
  return
}

} // end module

// -----

// Check that types are converted to 32-bit when no special capabilities that
// are not supported.
module attributes {
  spv.target_env = #spv.target_env<#spv.vce<v1.0, [], []>, {}>
} {

func @int_vector4_invalid(%arg0: vector<4xi64>) {
  // expected-error @+2 {{bitwidth emulation is not implemented yet on unsigned op}}
  // expected-error @+1 {{op requires the same type for all operands and results}}
  %0 = divi_unsigned %arg0, %arg0: vector<4xi64>
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
  %0 = and %arg0, %arg1 : i32
  // CHECK: spv.BitwiseOr
  %1 = or %arg0, %arg1 : i32
  // CHECK: spv.BitwiseXor
  %2 = xor %arg0, %arg1 : i32
  return
}

// CHECK-LABEL: @bitwise_vector
func @bitwise_vector(%arg0 : vector<4xi32>, %arg1 : vector<4xi32>) {
  // CHECK: spv.BitwiseAnd
  %0 = and %arg0, %arg1 : vector<4xi32>
  // CHECK: spv.BitwiseOr
  %1 = or %arg0, %arg1 : vector<4xi32>
  // CHECK: spv.BitwiseXor
  %2 = xor %arg0, %arg1 : vector<4xi32>
  return
}

// CHECK-LABEL: @logical_scalar
func @logical_scalar(%arg0 : i1, %arg1 : i1) {
  // CHECK: spv.LogicalAnd
  %0 = and %arg0, %arg1 : i1
  // CHECK: spv.LogicalOr
  %1 = or %arg0, %arg1 : i1
  return
}

// CHECK-LABEL: @logical_vector
func @logical_vector(%arg0 : vector<4xi1>, %arg1 : vector<4xi1>) {
  // CHECK: spv.LogicalAnd
  %0 = and %arg0, %arg1 : vector<4xi1>
  // CHECK: spv.LogicalOr
  %1 = or %arg0, %arg1 : vector<4xi1>
  return
}

// CHECK-LABEL: @shift_scalar
func @shift_scalar(%arg0 : i32, %arg1 : i32) {
  // CHECK: spv.ShiftLeftLogical
  %0 = shift_left %arg0, %arg1 : i32
  // CHECK: spv.ShiftRightArithmetic
  %1 = shift_right_signed %arg0, %arg1 : i32
  // CHECK: spv.ShiftRightLogical
  %2 = shift_right_unsigned %arg0, %arg1 : i32
  return
}

// CHECK-LABEL: @shift_vector
func @shift_vector(%arg0 : vector<4xi32>, %arg1 : vector<4xi32>) {
  // CHECK: spv.ShiftLeftLogical
  %0 = shift_left %arg0, %arg1 : vector<4xi32>
  // CHECK: spv.ShiftRightArithmetic
  %1 = shift_right_signed %arg0, %arg1 : vector<4xi32>
  // CHECK: spv.ShiftRightLogical
  %2 = shift_right_unsigned %arg0, %arg1 : vector<4xi32>
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
  %1 = cmpf oeq, %arg0, %arg1 : f32
  // CHECK: spv.FOrdGreaterThan
  %2 = cmpf ogt, %arg0, %arg1 : f32
  // CHECK: spv.FOrdGreaterThanEqual
  %3 = cmpf oge, %arg0, %arg1 : f32
  // CHECK: spv.FOrdLessThan
  %4 = cmpf olt, %arg0, %arg1 : f32
  // CHECK: spv.FOrdLessThanEqual
  %5 = cmpf ole, %arg0, %arg1 : f32
  // CHECK: spv.FOrdNotEqual
  %6 = cmpf one, %arg0, %arg1 : f32
  // CHECK: spv.FUnordEqual
  %7 = cmpf ueq, %arg0, %arg1 : f32
  // CHECK: spv.FUnordGreaterThan
  %8 = cmpf ugt, %arg0, %arg1 : f32
  // CHECK: spv.FUnordGreaterThanEqual
  %9 = cmpf uge, %arg0, %arg1 : f32
  // CHECK: spv.FUnordLessThan
  %10 = cmpf ult, %arg0, %arg1 : f32
  // CHECK: FUnordLessThanEqual
  %11 = cmpf ule, %arg0, %arg1 : f32
  // CHECK: spv.FUnordNotEqual
  %12 = cmpf une, %arg0, %arg1 : f32
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
  %0 = cmpf ord, %arg0, %arg1 : f32
  // CHECK: spv.Unordered
  %1 = cmpf uno, %arg0, %arg1 : f32
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
  %0 = cmpf ord, %arg0, %arg1 : f32

  // CHECK-NEXT: %[[LHS_NAN:.+]] = spv.IsNan %[[LHS]] : f32
  // CHECK-NEXT: %[[RHS_NAN:.+]] = spv.IsNan %[[RHS]] : f32
  // CHECK-NEXT: %{{.+}} = spv.LogicalOr %[[LHS_NAN]], %[[RHS_NAN]] : i1
  %1 = cmpf uno, %arg0, %arg1 : f32
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
  %0 = cmpi eq, %arg0, %arg1 : i32
  // CHECK: spv.INotEqual
  %1 = cmpi ne, %arg0, %arg1 : i32
  // CHECK: spv.SLessThan
  %2 = cmpi slt, %arg0, %arg1 : i32
  // CHECK: spv.SLessThanEqual
  %3 = cmpi sle, %arg0, %arg1 : i32
  // CHECK: spv.SGreaterThan
  %4 = cmpi sgt, %arg0, %arg1 : i32
  // CHECK: spv.SGreaterThanEqual
  %5 = cmpi sge, %arg0, %arg1 : i32
  // CHECK: spv.ULessThan
  %6 = cmpi ult, %arg0, %arg1 : i32
  // CHECK: spv.ULessThanEqual
  %7 = cmpi ule, %arg0, %arg1 : i32
  // CHECK: spv.UGreaterThan
  %8 = cmpi ugt, %arg0, %arg1 : i32
  // CHECK: spv.UGreaterThanEqual
  %9 = cmpi uge, %arg0, %arg1 : i32
  return
}

// CHECK-LABEL: @boolcmpi
func @boolcmpi(%arg0 : i1, %arg1 : i1) {
  // CHECK: spv.LogicalEqual
  %0 = cmpi eq, %arg0, %arg1 : i1
  // CHECK: spv.LogicalNotEqual
  %1 = cmpi ne, %arg0, %arg1 : i1
  return
}

// CHECK-LABEL: @vecboolcmpi
func @vecboolcmpi(%arg0 : vector<4xi1>, %arg1 : vector<4xi1>) {
  // CHECK: spv.LogicalEqual
  %0 = cmpi eq, %arg0, %arg1 : vector<4xi1>
  // CHECK: spv.LogicalNotEqual
  %1 = cmpi ne, %arg0, %arg1 : vector<4xi1>
  return
}

} // end module

// -----

//===----------------------------------------------------------------------===//
// std.constant
//===----------------------------------------------------------------------===//

module attributes {
  spv.target_env = #spv.target_env<
    #spv.vce<v1.0, [Int8, Int16, Int64, Float16, Float64], []>, {}>
} {

// CHECK-LABEL: @constant
func @constant() {
  // CHECK: spv.Constant true
  %0 = constant true
  // CHECK: spv.Constant 42 : i32
  %1 = constant 42 : i32
  // CHECK: spv.Constant 5.000000e-01 : f32
  %2 = constant 0.5 : f32
  // CHECK: spv.Constant dense<[2, 3]> : vector<2xi32>
  %3 = constant dense<[2, 3]> : vector<2xi32>
  // CHECK: spv.Constant 1 : i32
  %4 = constant 1 : index
  // CHECK: spv.Constant dense<1> : tensor<6xi32> : !spv.array<6 x i32, stride=4>
  %5 = constant dense<1> : tensor<2x3xi32>
  // CHECK: spv.Constant dense<1.000000e+00> : tensor<6xf32> : !spv.array<6 x f32, stride=4>
  %6 = constant dense<1.0> : tensor<2x3xf32>
  // CHECK: spv.Constant dense<{{\[}}1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00]> : tensor<6xf32> : !spv.array<6 x f32, stride=4>
  %7 = constant dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf32>
  // CHECK: spv.Constant dense<{{\[}}1, 2, 3, 4, 5, 6]> : tensor<6xi32> : !spv.array<6 x i32, stride=4>
  %8 = constant dense<[[1, 2, 3], [4, 5, 6]]> : tensor<2x3xi32>
  // CHECK: spv.Constant dense<{{\[}}1, 2, 3, 4, 5, 6]> : tensor<6xi32> : !spv.array<6 x i32, stride=4>
  %9 =  constant dense<[[1, 2], [3, 4], [5, 6]]> : tensor<3x2xi32>
  // CHECK: spv.Constant dense<{{\[}}1, 2, 3, 4, 5, 6]> : tensor<6xi32> : !spv.array<6 x i32, stride=4>
  %10 =  constant dense<[1, 2, 3, 4, 5, 6]> : tensor<6xi32>
  return
}

// CHECK-LABEL: @constant_16bit
func @constant_16bit() {
  // CHECK: spv.Constant 4 : i16
  %0 = constant 4 : i16
  // CHECK: spv.Constant 5.000000e+00 : f16
  %1 = constant 5.0 : f16
  // CHECK: spv.Constant dense<[2, 3]> : vector<2xi16>
  %2 = constant dense<[2, 3]> : vector<2xi16>
  // CHECK: spv.Constant dense<4.000000e+00> : tensor<5xf16> : !spv.array<5 x f16, stride=2>
  %3 = constant dense<4.0> : tensor<5xf16>
  return
}

// CHECK-LABEL: @constant_64bit
func @constant_64bit() {
  // CHECK: spv.Constant 4 : i64
  %0 = constant 4 : i64
  // CHECK: spv.Constant 5.000000e+00 : f64
  %1 = constant 5.0 : f64
  // CHECK: spv.Constant dense<[2, 3]> : vector<2xi64>
  %2 = constant dense<[2, 3]> : vector<2xi64>
  // CHECK: spv.Constant dense<4.000000e+00> : tensor<5xf64> : !spv.array<5 x f64, stride=8>
  %3 = constant dense<4.0> : tensor<5xf64>
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
  %0 = constant 4 : i16
  // CHECK: spv.Constant 5.000000e+00 : f32
  %1 = constant 5.0 : f16
  // CHECK: spv.Constant dense<[2, 3]> : vector<2xi32>
  %2 = constant dense<[2, 3]> : vector<2xi16>
  // CHECK: spv.Constant dense<4.000000e+00> : tensor<5xf32> : !spv.array<5 x f32, stride=4>
  %3 = constant dense<4.0> : tensor<5xf16>
  // CHECK: spv.Constant dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00]> : tensor<4xf32> : !spv.array<4 x f32, stride=4>
  %4 = constant dense<[[1.0, 2.0], [3.0, 4.0]]> : tensor<2x2xf16>
  return
}

// CHECK-LABEL: @constant_64bit
func @constant_64bit() {
  // CHECK: spv.Constant 4 : i32
  %0 = constant 4 : i64
  // CHECK: spv.Constant 5.000000e+00 : f32
  %1 = constant 5.0 : f64
  // CHECK: spv.Constant dense<[2, 3]> : vector<2xi32>
  %2 = constant dense<[2, 3]> : vector<2xi64>
  // CHECK: spv.Constant dense<4.000000e+00> : tensor<5xf32> : !spv.array<5 x f32, stride=4>
  %3 = constant dense<4.0> : tensor<5xf64>
  // CHECK: spv.Constant dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00]> : tensor<4xf32> : !spv.array<4 x f32, stride=4>
  %4 = constant dense<[[1.0, 2.0], [3.0, 4.0]]> : tensor<2x2xf16>
  return
}

// CHECK-LABEL: @corner_cases
func @corner_cases() {
  // CHECK: %{{.*}} = spv.Constant -1 : i32
  %0 = constant 4294967295  : i64 // 2^32 - 1
  // CHECK: %{{.*}} = spv.Constant 2147483647 : i32
  %1 = constant 2147483647  : i64 // 2^31 - 1
  // CHECK: %{{.*}} = spv.Constant -2147483648 : i32
  %2 = constant 2147483648  : i64 // 2^31
  // CHECK: %{{.*}} = spv.Constant -2147483648 : i32
  %3 = constant -2147483648 : i64 // -2^31

  // CHECK: %{{.*}} = spv.Constant -1 : i32
  %5 = constant -1 : i64
  // CHECK: %{{.*}} = spv.Constant -2 : i32
  %6 = constant -2 : i64
  // CHECK: %{{.*}} = spv.Constant -1 : i32
  %7 = constant -1 : index
  // CHECK: %{{.*}} = spv.Constant -2 : i32
  %8 = constant -2 : index


  // CHECK: spv.Constant false
  %9 = constant false
  // CHECK: spv.Constant true
  %10 = constant true

  return
}

// CHECK-LABEL: @unsupported_cases
func @unsupported_cases() {
  // CHECK: %{{.*}} = constant 4294967296 : i64
  %0 = constant 4294967296 : i64 // 2^32
  // CHECK: %{{.*}} = constant -2147483649 : i64
  %1 = constant -2147483649 : i64 // -2^31 - 1
  // CHECK: %{{.*}} = constant 1.0000000000000002 : f64
  %2 = constant 0x3FF0000000000001 : f64 // smallest number > 1
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
  %0 = index_cast %arg0 : i16 to index
  return
}

// CHECK-LABEL: index_cast2
func @index_cast2(%arg0: index) {
  // CHECK: spv.SConvert %{{.+}} : i32 to i16
  %0 = index_cast %arg0 : index to i16
  return
}

// CHECK-LABEL: index_cast3
func @index_cast3(%arg0: i32) {
  // CHECK-NOT: spv.SConvert
  %0 = index_cast %arg0 : i32 to index
  return
}

// CHECK-LABEL: index_cast4
func @index_cast4(%arg0: index) {
  // CHECK-NOT: spv.SConvert
  %0 = index_cast %arg0 : index to i32
  return
}

// CHECK-LABEL: @fpext1
func @fpext1(%arg0: f16) -> f64 {
  // CHECK: spv.FConvert %{{.*}} : f16 to f64
  %0 = std.fpext %arg0 : f16 to f64
  return %0 : f64
}

// CHECK-LABEL: @fpext2
func @fpext2(%arg0 : f32) -> f64 {
  // CHECK: spv.FConvert %{{.*}} : f32 to f64
  %0 = std.fpext %arg0 : f32 to f64
  return %0 : f64
}

// CHECK-LABEL: @fptrunc1
func @fptrunc1(%arg0 : f64) -> f16 {
  // CHECK: spv.FConvert %{{.*}} : f64 to f16
  %0 = std.fptrunc %arg0 : f64 to f16
  return %0 : f16
}

// CHECK-LABEL: @fptrunc2
func @fptrunc2(%arg0: f32) -> f16 {
  // CHECK: spv.FConvert %{{.*}} : f32 to f16
  %0 = std.fptrunc %arg0 : f32 to f16
  return %0 : f16
}

// CHECK-LABEL: @sitofp1
func @sitofp1(%arg0 : i32) -> f32 {
  // CHECK: spv.ConvertSToF %{{.*}} : i32 to f32
  %0 = std.sitofp %arg0 : i32 to f32
  return %0 : f32
}

// CHECK-LABEL: @sitofp2
func @sitofp2(%arg0 : i64) -> f64 {
  // CHECK: spv.ConvertSToF %{{.*}} : i64 to f64
  %0 = std.sitofp %arg0 : i64 to f64
  return %0 : f64
}

// CHECK-LABEL: @uitofp_i16_f32
func @uitofp_i16_f32(%arg0: i16) -> f32 {
  // CHECK: spv.ConvertUToF %{{.*}} : i16 to f32
  %0 = std.uitofp %arg0 : i16 to f32
  return %0 : f32
}

// CHECK-LABEL: @uitofp_i32_f32
func @uitofp_i32_f32(%arg0 : i32) -> f32 {
  // CHECK: spv.ConvertUToF %{{.*}} : i32 to f32
  %0 = std.uitofp %arg0 : i32 to f32
  return %0 : f32
}

// CHECK-LABEL: @uitofp_i1_f32
func @uitofp_i1_f32(%arg0 : i1) -> f32 {
  // CHECK: %[[ZERO:.+]] = spv.Constant 0.000000e+00 : f32
  // CHECK: %[[ONE:.+]] = spv.Constant 1.000000e+00 : f32
  // CHECK: spv.Select %{{.*}}, %[[ONE]], %[[ZERO]] : i1, f32
  %0 = std.uitofp %arg0 : i1 to f32
  return %0 : f32
}

// CHECK-LABEL: @uitofp_i1_f64
func @uitofp_i1_f64(%arg0 : i1) -> f64 {
  // CHECK: %[[ZERO:.+]] = spv.Constant 0.000000e+00 : f64
  // CHECK: %[[ONE:.+]] = spv.Constant 1.000000e+00 : f64
  // CHECK: spv.Select %{{.*}}, %[[ONE]], %[[ZERO]] : i1, f64
  %0 = std.uitofp %arg0 : i1 to f64
  return %0 : f64
}

// CHECK-LABEL: @uitofp_vec_i1_f32
func @uitofp_vec_i1_f32(%arg0 : vector<4xi1>) -> vector<4xf32> {
  // CHECK: %[[ZERO:.+]] = spv.Constant dense<0.000000e+00> : vector<4xf32>
  // CHECK: %[[ONE:.+]] = spv.Constant dense<1.000000e+00> : vector<4xf32>
  // CHECK: spv.Select %{{.*}}, %[[ONE]], %[[ZERO]] : vector<4xi1>, vector<4xf32>
  %0 = std.uitofp %arg0 : vector<4xi1> to vector<4xf32>
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
  %0 = std.sexti %arg0 : i16 to i64
  return %0 : i64
}

// CHECK-LABEL: @sexti2
func @sexti2(%arg0 : i32) -> i64 {
  // CHECK: spv.SConvert %{{.*}} : i32 to i64
  %0 = std.sexti %arg0 : i32 to i64
  return %0 : i64
}

// CHECK-LABEL: @zexti1
func @zexti1(%arg0: i16) -> i64 {
  // CHECK: spv.UConvert %{{.*}} : i16 to i64
  %0 = std.zexti %arg0 : i16 to i64
  return %0 : i64
}

// CHECK-LABEL: @zexti2
func @zexti2(%arg0 : i32) -> i64 {
  // CHECK: spv.UConvert %{{.*}} : i32 to i64
  %0 = std.zexti %arg0 : i32 to i64
  return %0 : i64
}

// CHECK-LABEL: @zexti3
func @zexti3(%arg0 : i1) -> i32 {
  // CHECK: %[[ZERO:.+]] = spv.Constant 0 : i32
  // CHECK: %[[ONE:.+]] = spv.Constant 1 : i32
  // CHECK: spv.Select %{{.*}}, %[[ONE]], %[[ZERO]] : i1, i32
  %0 = std.zexti %arg0 : i1 to i32
  return %0 : i32
}

// CHECK-LABEL: @zexti4
func @zexti4(%arg0 : vector<4xi1>) -> vector<4xi32> {
  // CHECK: %[[ZERO:.+]] = spv.Constant dense<0> : vector<4xi32>
  // CHECK: %[[ONE:.+]] = spv.Constant dense<1> : vector<4xi32>
  // CHECK: spv.Select %{{.*}}, %[[ONE]], %[[ZERO]] : vector<4xi1>, vector<4xi32>
  %0 = std.zexti %arg0 : vector<4xi1> to vector<4xi32>
  return %0 : vector<4xi32>
}

// CHECK-LABEL: @zexti5
func @zexti5(%arg0 : vector<4xi1>) -> vector<4xi64> {
  // CHECK: %[[ZERO:.+]] = spv.Constant dense<0> : vector<4xi64>
  // CHECK: %[[ONE:.+]] = spv.Constant dense<1> : vector<4xi64>
  // CHECK: spv.Select %{{.*}}, %[[ONE]], %[[ZERO]] : vector<4xi1>, vector<4xi64>
  %0 = std.zexti %arg0 : vector<4xi1> to vector<4xi64>
  return %0 : vector<4xi64>
}

// CHECK-LABEL: @trunci1
func @trunci1(%arg0 : i64) -> i16 {
  // CHECK: spv.SConvert %{{.*}} : i64 to i16
  %0 = std.trunci %arg0 : i64 to i16
  return %0 : i16
}

// CHECK-LABEL: @trunci2
func @trunci2(%arg0: i32) -> i16 {
  // CHECK: spv.SConvert %{{.*}} : i32 to i16
  %0 = std.trunci %arg0 : i32 to i16
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
  %0 = std.trunci %arg0 : i32 to i1
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
  %0 = std.trunci %arg0 : vector<4xi32> to vector<4xi1>
  return %0 : vector<4xi1>
}

// CHECK-LABEL: @fptosi1
func @fptosi1(%arg0 : f32) -> i32 {
  // CHECK: spv.ConvertFToS %{{.*}} : f32 to i32
  %0 = std.fptosi %arg0 : f32 to i32
  return %0 : i32
}

// CHECK-LABEL: @fptosi2
func @fptosi2(%arg0 : f16) -> i16 {
  // CHECK: spv.ConvertFToS %{{.*}} : f16 to i16
  %0 = std.fptosi %arg0 : f16 to i16
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
  %0 = std.fpext %arg0 : f16 to f64
  return %0: f64
}

// CHECK-LABEL: @fpext2
// CHECK-SAME: %[[ARG:.*]]: f32
func @fpext2(%arg0 : f32) -> f64 {
  // CHECK-NEXT: spv.FConvert %[[ARG]] : f32 to f64
  %0 = std.fpext %arg0 : f32 to f64
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
  %0 = std.fptrunc %arg0 : f64 to f16
  return %0: f16
}

// CHECK-LABEL: @fptrunc2
// CHECK-SAME: %[[ARG:.*]]: f32
func @fptrunc2(%arg0: f32) -> f16 {
  // CHECK-NEXT: spv.FConvert %[[ARG]] : f32 to f16
  %0 = std.fptrunc %arg0 : f32 to f16
  return %0: f16
}

// CHECK-LABEL: @sitofp
func @sitofp(%arg0 : i64) -> f64 {
  // CHECK: spv.ConvertSToF %{{.*}} : i32 to f32
  %0 = std.sitofp %arg0 : i64 to f64
  return %0: f64
}

} // end module

// -----

module attributes {
  spv.target_env = #spv.target_env<
    #spv.vce<v1.0, [Shader, Int8, Int16, Int64, Float16, Float64],
             [SPV_KHR_storage_buffer_storage_class]>, {}>
} {

//===----------------------------------------------------------------------===//
// std.select
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @select
func @select(%arg0 : i32, %arg1 : i32) {
  %0 = cmpi sle, %arg0, %arg1 : i32
  // CHECK: spv.Select
  %1 = select %0, %arg0, %arg1 : i32
  return
}

//===----------------------------------------------------------------------===//
// std load/store ops
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @load_store_zero_rank_float
// CHECK: [[ARG0:%.*]]: !spv.ptr<!spv.struct<(!spv.array<1 x f32, stride=4> [0])>, StorageBuffer>,
// CHECK: [[ARG1:%.*]]: !spv.ptr<!spv.struct<(!spv.array<1 x f32, stride=4> [0])>, StorageBuffer>)
func @load_store_zero_rank_float(%arg0: memref<f32>, %arg1: memref<f32>) {
  //      CHECK: [[ZERO1:%.*]] = spv.Constant 0 : i32
  //      CHECK: spv.AccessChain [[ARG0]][
  // CHECK-SAME: [[ZERO1]], [[ZERO1]]
  // CHECK-SAME: ] :
  //      CHECK: spv.Load "StorageBuffer" %{{.*}} : f32
  %0 = load %arg0[] : memref<f32>
  //      CHECK: [[ZERO2:%.*]] = spv.Constant 0 : i32
  //      CHECK: spv.AccessChain [[ARG1]][
  // CHECK-SAME: [[ZERO2]], [[ZERO2]]
  // CHECK-SAME: ] :
  //      CHECK: spv.Store "StorageBuffer" %{{.*}} : f32
  store %0, %arg1[] : memref<f32>
  return
}

// CHECK-LABEL: @load_store_zero_rank_int
// CHECK: [[ARG0:%.*]]: !spv.ptr<!spv.struct<(!spv.array<1 x i32, stride=4> [0])>, StorageBuffer>,
// CHECK: [[ARG1:%.*]]: !spv.ptr<!spv.struct<(!spv.array<1 x i32, stride=4> [0])>, StorageBuffer>)
func @load_store_zero_rank_int(%arg0: memref<i32>, %arg1: memref<i32>) {
  //      CHECK: [[ZERO1:%.*]] = spv.Constant 0 : i32
  //      CHECK: spv.AccessChain [[ARG0]][
  // CHECK-SAME: [[ZERO1]], [[ZERO1]]
  // CHECK-SAME: ] :
  //      CHECK: spv.Load "StorageBuffer" %{{.*}} : i32
  %0 = load %arg0[] : memref<i32>
  //      CHECK: [[ZERO2:%.*]] = spv.Constant 0 : i32
  //      CHECK: spv.AccessChain [[ARG1]][
  // CHECK-SAME: [[ZERO2]], [[ZERO2]]
  // CHECK-SAME: ] :
  //      CHECK: spv.Store "StorageBuffer" %{{.*}} : i32
  store %0, %arg1[] : memref<i32>
  return
}

} // end module

// -----

// Check that access chain indices are properly adjusted if non-32-bit types are
// emulated via 32-bit types.
// TODO: Test i1 and i64 types.
module attributes {
  spv.target_env = #spv.target_env<
    #spv.vce<v1.0, [Shader], [SPV_KHR_storage_buffer_storage_class]>, {}>
} {

// CHECK-LABEL: @load_i8
func @load_i8(%arg0: memref<i8>) {
  //     CHECK: %[[ZERO:.+]] = spv.Constant 0 : i32
  //     CHECK: %[[FOUR1:.+]] = spv.Constant 4 : i32
  //     CHECK: %[[QUOTIENT:.+]] = spv.SDiv %[[ZERO]], %[[FOUR1]] : i32
  //     CHECK: %[[PTR:.+]] = spv.AccessChain %{{.+}}[%[[ZERO]], %[[QUOTIENT]]]
  //     CHECK: %[[LOAD:.+]] = spv.Load  "StorageBuffer" %[[PTR]]
  //     CHECK: %[[FOUR2:.+]] = spv.Constant 4 : i32
  //     CHECK: %[[EIGHT:.+]] = spv.Constant 8 : i32
  //     CHECK: %[[IDX:.+]] = spv.UMod %[[ZERO]], %[[FOUR2]] : i32
  //     CHECK: %[[BITS:.+]] = spv.IMul %[[IDX]], %[[EIGHT]] : i32
  //     CHECK: %[[VALUE:.+]] = spv.ShiftRightArithmetic %[[LOAD]], %[[BITS]] : i32, i32
  //     CHECK: %[[MASK:.+]] = spv.Constant 255 : i32
  //     CHECK: %[[T1:.+]] = spv.BitwiseAnd %[[VALUE]], %[[MASK]] : i32
  //     CHECK: %[[T2:.+]] = spv.Constant 24 : i32
  //     CHECK: %[[T3:.+]] = spv.ShiftLeftLogical %[[T1]], %[[T2]] : i32, i32
  //     CHECK: spv.ShiftRightArithmetic %[[T3]], %[[T2]] : i32, i32
  %0 = load %arg0[] : memref<i8>
  return
}

// CHECK-LABEL: @load_i16
//       CHECK: (%[[ARG0:.+]]: {{.*}}, %[[ARG1:.+]]: i32)
func @load_i16(%arg0: memref<10xi16>, %index : index) {
  //     CHECK: %[[ZERO:.+]] = spv.Constant 0 : i32
  //     CHECK: %[[OFFSET:.+]] = spv.Constant 0 : i32
  //     CHECK: %[[ONE:.+]] = spv.Constant 1 : i32
  //     CHECK: %[[UPDATE:.+]] = spv.IMul %[[ONE]], %[[ARG1]] : i32
  //     CHECK: %[[FLAT_IDX:.+]] = spv.IAdd %[[OFFSET]], %[[UPDATE]] : i32
  //     CHECK: %[[TWO1:.+]] = spv.Constant 2 : i32
  //     CHECK: %[[QUOTIENT:.+]] = spv.SDiv %[[FLAT_IDX]], %[[TWO1]] : i32
  //     CHECK: %[[PTR:.+]] = spv.AccessChain %{{.+}}[%[[ZERO]], %[[QUOTIENT]]]
  //     CHECK: %[[LOAD:.+]] = spv.Load  "StorageBuffer" %[[PTR]]
  //     CHECK: %[[TWO2:.+]] = spv.Constant 2 : i32
  //     CHECK: %[[SIXTEEN:.+]] = spv.Constant 16 : i32
  //     CHECK: %[[IDX:.+]] = spv.UMod %[[FLAT_IDX]], %[[TWO2]] : i32
  //     CHECK: %[[BITS:.+]] = spv.IMul %[[IDX]], %[[SIXTEEN]] : i32
  //     CHECK: %[[VALUE:.+]] = spv.ShiftRightArithmetic %[[LOAD]], %[[BITS]] : i32, i32
  //     CHECK: %[[MASK:.+]] = spv.Constant 65535 : i32
  //     CHECK: %[[T1:.+]] = spv.BitwiseAnd %[[VALUE]], %[[MASK]] : i32
  //     CHECK: %[[T2:.+]] = spv.Constant 16 : i32
  //     CHECK: %[[T3:.+]] = spv.ShiftLeftLogical %[[T1]], %[[T2]] : i32, i32
  //     CHECK: spv.ShiftRightArithmetic %[[T3]], %[[T2]] : i32, i32
  %0 = load %arg0[%index] : memref<10xi16>
  return
}

// CHECK-LABEL: @load_i32
func @load_i32(%arg0: memref<i32>) {
  // CHECK-NOT: spv.SDiv
  //     CHECK: spv.Load
  // CHECK-NOT: spv.ShiftRightArithmetic
  %0 = load %arg0[] : memref<i32>
  return
}

// CHECK-LABEL: @load_f32
func @load_f32(%arg0: memref<f32>) {
  // CHECK-NOT: spv.SDiv
  //     CHECK: spv.Load
  // CHECK-NOT: spv.ShiftRightArithmetic
  %0 = load %arg0[] : memref<f32>
  return
}

// CHECK-LABEL: @store_i8
//       CHECK: (%[[ARG0:.+]]: {{.*}}, %[[ARG1:.+]]: i32)
func @store_i8(%arg0: memref<i8>, %value: i8) {
  //     CHECK: %[[ZERO:.+]] = spv.Constant 0 : i32
  //     CHECK: %[[FOUR:.+]] = spv.Constant 4 : i32
  //     CHECK: %[[EIGHT:.+]] = spv.Constant 8 : i32
  //     CHECK: %[[IDX:.+]] = spv.UMod %[[ZERO]], %[[FOUR]] : i32
  //     CHECK: %[[OFFSET:.+]] = spv.IMul %[[IDX]], %[[EIGHT]] : i32
  //     CHECK: %[[MASK1:.+]] = spv.Constant 255 : i32
  //     CHECK: %[[TMP1:.+]] = spv.ShiftLeftLogical %[[MASK1]], %[[OFFSET]] : i32, i32
  //     CHECK: %[[MASK:.+]] = spv.Not %[[TMP1]] : i32
  //     CHECK: %[[CLAMPED_VAL:.+]] = spv.BitwiseAnd %[[ARG1]], %[[MASK1]] : i32
  //     CHECK: %[[STORE_VAL:.+]] = spv.ShiftLeftLogical %[[CLAMPED_VAL]], %[[OFFSET]] : i32, i32
  //     CHECK: %[[FOUR2:.+]] = spv.Constant 4 : i32
  //     CHECK: %[[ACCESS_IDX:.+]] = spv.SDiv %[[ZERO]], %[[FOUR2]] : i32
  //     CHECK: %[[PTR:.+]] = spv.AccessChain %[[ARG0]][%[[ZERO]], %[[ACCESS_IDX]]]
  //     CHECK: spv.AtomicAnd "Device" "AcquireRelease" %[[PTR]], %[[MASK]]
  //     CHECK: spv.AtomicOr "Device" "AcquireRelease" %[[PTR]], %[[STORE_VAL]]
  store %value, %arg0[] : memref<i8>
  return
}

// CHECK-LABEL: @store_i16
//       CHECK: (%[[ARG0:.+]]: {{.*}}, %[[ARG1:.+]]: i32, %[[ARG2:.+]]: i32)
func @store_i16(%arg0: memref<10xi16>, %index: index, %value: i16) {
  //     CHECK: %[[ZERO:.+]] = spv.Constant 0 : i32
  //     CHECK: %[[OFFSET:.+]] = spv.Constant 0 : i32
  //     CHECK: %[[ONE:.+]] = spv.Constant 1 : i32
  //     CHECK: %[[UPDATE:.+]] = spv.IMul %[[ONE]], %[[ARG1]] : i32
  //     CHECK: %[[FLAT_IDX:.+]] = spv.IAdd %[[OFFSET]], %[[UPDATE]] : i32
  //     CHECK: %[[TWO:.+]] = spv.Constant 2 : i32
  //     CHECK: %[[SIXTEEN:.+]] = spv.Constant 16 : i32
  //     CHECK: %[[IDX:.+]] = spv.UMod %[[FLAT_IDX]], %[[TWO]] : i32
  //     CHECK: %[[OFFSET:.+]] = spv.IMul %[[IDX]], %[[SIXTEEN]] : i32
  //     CHECK: %[[MASK1:.+]] = spv.Constant 65535 : i32
  //     CHECK: %[[TMP1:.+]] = spv.ShiftLeftLogical %[[MASK1]], %[[OFFSET]] : i32, i32
  //     CHECK: %[[MASK:.+]] = spv.Not %[[TMP1]] : i32
  //     CHECK: %[[CLAMPED_VAL:.+]] = spv.BitwiseAnd %[[ARG2]], %[[MASK1]] : i32
  //     CHECK: %[[STORE_VAL:.+]] = spv.ShiftLeftLogical %[[CLAMPED_VAL]], %[[OFFSET]] : i32, i32
  //     CHECK: %[[TWO2:.+]] = spv.Constant 2 : i32
  //     CHECK: %[[ACCESS_IDX:.+]] = spv.SDiv %[[FLAT_IDX]], %[[TWO2]] : i32
  //     CHECK: %[[PTR:.+]] = spv.AccessChain %[[ARG0]][%[[ZERO]], %[[ACCESS_IDX]]]
  //     CHECK: spv.AtomicAnd "Device" "AcquireRelease" %[[PTR]], %[[MASK]]
  //     CHECK: spv.AtomicOr "Device" "AcquireRelease" %[[PTR]], %[[STORE_VAL]]
  store %value, %arg0[%index] : memref<10xi16>
  return
}

// CHECK-LABEL: @store_i32
func @store_i32(%arg0: memref<i32>, %value: i32) {
  //     CHECK: spv.Store
  // CHECK-NOT: spv.AtomicAnd
  // CHECK-NOT: spv.AtomicOr
  store %value, %arg0[] : memref<i32>
  return
}

// CHECK-LABEL: @store_f32
func @store_f32(%arg0: memref<f32>, %value: f32) {
  //     CHECK: spv.Store
  // CHECK-NOT: spv.AtomicAnd
  // CHECK-NOT: spv.AtomicOr
  store %value, %arg0[] : memref<f32>
  return
}

} // end module

// -----

// Check that access chain indices are properly adjusted if non-16/32-bit types
// are emulated via 32-bit types.
module attributes {
  spv.target_env = #spv.target_env<
    #spv.vce<v1.0, [Int16, StorageBuffer16BitAccess, Shader],
    [SPV_KHR_storage_buffer_storage_class, SPV_KHR_16bit_storage]>, {}>
} {

// CHECK-LABEL: @load_i8
func @load_i8(%arg0: memref<i8>) {
  //     CHECK: %[[ZERO:.+]] = spv.Constant 0 : i32
  //     CHECK: %[[FOUR1:.+]] = spv.Constant 4 : i32
  //     CHECK: %[[QUOTIENT:.+]] = spv.SDiv %[[ZERO]], %[[FOUR1]] : i32
  //     CHECK: %[[PTR:.+]] = spv.AccessChain %{{.+}}[%[[ZERO]], %[[QUOTIENT]]]
  //     CHECK: %[[LOAD:.+]] = spv.Load  "StorageBuffer" %[[PTR]]
  //     CHECK: %[[FOUR2:.+]] = spv.Constant 4 : i32
  //     CHECK: %[[EIGHT:.+]] = spv.Constant 8 : i32
  //     CHECK: %[[IDX:.+]] = spv.UMod %[[ZERO]], %[[FOUR2]] : i32
  //     CHECK: %[[BITS:.+]] = spv.IMul %[[IDX]], %[[EIGHT]] : i32
  //     CHECK: %[[VALUE:.+]] = spv.ShiftRightArithmetic %[[LOAD]], %[[BITS]] : i32, i32
  //     CHECK: %[[MASK:.+]] = spv.Constant 255 : i32
  //     CHECK: %[[T1:.+]] = spv.BitwiseAnd %[[VALUE]], %[[MASK]] : i32
  //     CHECK: %[[T2:.+]] = spv.Constant 24 : i32
  //     CHECK: %[[T3:.+]] = spv.ShiftLeftLogical %[[T1]], %[[T2]] : i32, i32
  //     CHECK: spv.ShiftRightArithmetic %[[T3]], %[[T2]] : i32, i32
  %0 = load %arg0[] : memref<i8>
  return
}

// CHECK-LABEL: @load_i16
func @load_i16(%arg0: memref<i16>) {
  // CHECK-NOT: spv.SDiv
  //     CHECK: spv.Load
  // CHECK-NOT: spv.ShiftRightArithmetic
  %0 = load %arg0[] : memref<i16>
  return
}

// CHECK-LABEL: @store_i8
//       CHECK: (%[[ARG0:.+]]: {{.*}}, %[[ARG1:.+]]: i32)
func @store_i8(%arg0: memref<i8>, %value: i8) {
  //     CHECK: %[[ZERO:.+]] = spv.Constant 0 : i32
  //     CHECK: %[[FOUR:.+]] = spv.Constant 4 : i32
  //     CHECK: %[[EIGHT:.+]] = spv.Constant 8 : i32
  //     CHECK: %[[IDX:.+]] = spv.UMod %[[ZERO]], %[[FOUR]] : i32
  //     CHECK: %[[OFFSET:.+]] = spv.IMul %[[IDX]], %[[EIGHT]] : i32
  //     CHECK: %[[MASK1:.+]] = spv.Constant 255 : i32
  //     CHECK: %[[TMP1:.+]] = spv.ShiftLeftLogical %[[MASK1]], %[[OFFSET]] : i32, i32
  //     CHECK: %[[MASK:.+]] = spv.Not %[[TMP1]] : i32
  //     CHECK: %[[CLAMPED_VAL:.+]] = spv.BitwiseAnd %[[ARG1]], %[[MASK1]] : i32
  //     CHECK: %[[STORE_VAL:.+]] = spv.ShiftLeftLogical %[[CLAMPED_VAL]], %[[OFFSET]] : i32, i32
  //     CHECK: %[[FOUR2:.+]] = spv.Constant 4 : i32
  //     CHECK: %[[ACCESS_IDX:.+]] = spv.SDiv %[[ZERO]], %[[FOUR2]] : i32
  //     CHECK: %[[PTR:.+]] = spv.AccessChain %[[ARG0]][%[[ZERO]], %[[ACCESS_IDX]]]
  //     CHECK: spv.AtomicAnd "Device" "AcquireRelease" %[[PTR]], %[[MASK]]
  //     CHECK: spv.AtomicOr "Device" "AcquireRelease" %[[PTR]], %[[STORE_VAL]]
  store %value, %arg0[] : memref<i8>
  return
}

// CHECK-LABEL: @store_i16
func @store_i16(%arg0: memref<10xi16>, %index: index, %value: i16) {
  //     CHECK: spv.Store
  // CHECK-NOT: spv.AtomicAnd
  // CHECK-NOT: spv.AtomicOr
  store %value, %arg0[%index] : memref<10xi16>
  return
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
