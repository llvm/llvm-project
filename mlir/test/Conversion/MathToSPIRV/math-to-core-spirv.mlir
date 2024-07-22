// RUN: mlir-opt -split-input-file -convert-math-to-spirv -verify-diagnostics %s -o - | FileCheck %s

func.func @copy_sign_scalar(%value: f32, %sign: f32) -> f32 {
  %0 = math.copysign %value, %sign : f32
  return %0: f32
}

// CHECK-LABEL: func @copy_sign_scalar
//  CHECK-SAME: (%[[VALUE:.+]]: f32, %[[SIGN:.+]]: f32)
//       CHECK:   %[[SMASK:.+]] = spirv.Constant -2147483648 : i32
//       CHECK:   %[[VMASK:.+]] = spirv.Constant 2147483647 : i32
//       CHECK:   %[[VCAST:.+]] = spirv.Bitcast %[[VALUE]] : f32 to i32
//       CHECK:   %[[SCAST:.+]] = spirv.Bitcast %[[SIGN]] : f32 to i32
//       CHECK:   %[[VAND:.+]] = spirv.BitwiseAnd %[[VCAST]], %[[VMASK]] : i32
//       CHECK:   %[[SAND:.+]] = spirv.BitwiseAnd %[[SCAST]], %[[SMASK]] : i32
//       CHECK:   %[[OR:.+]] = spirv.BitwiseOr %[[VAND]], %[[SAND]] : i32
//       CHECK:   %[[RESULT:.+]] = spirv.Bitcast %[[OR]] : i32 to f32
//       CHECK:   return %[[RESULT]]

// -----

module attributes { spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Float16, Int16], []>, #spirv.resource_limits<>> } {

func.func @copy_sign_vector(%value: vector<3xf16>, %sign: vector<3xf16>) -> vector<3xf16> {
  %0 = math.copysign %value, %sign : vector<3xf16>
  return %0: vector<3xf16>
}

}

// CHECK-LABEL: func @copy_sign_vector
//  CHECK-SAME: (%[[VALUE:.+]]: vector<3xf16>, %[[SIGN:.+]]: vector<3xf16>)
//       CHECK:   %[[SMASK:.+]] = spirv.Constant -32768 : i16
//       CHECK:   %[[VMASK:.+]] = spirv.Constant 32767 : i16
//       CHECK:   %[[SVMASK:.+]] = spirv.CompositeConstruct %[[SMASK]], %[[SMASK]], %[[SMASK]]
//       CHECK:   %[[VVMASK:.+]] = spirv.CompositeConstruct %[[VMASK]], %[[VMASK]], %[[VMASK]]
//       CHECK:   %[[VCAST:.+]] = spirv.Bitcast %[[VALUE]] : vector<3xf16> to vector<3xi16>
//       CHECK:   %[[SCAST:.+]] = spirv.Bitcast %[[SIGN]] : vector<3xf16> to vector<3xi16>
//       CHECK:   %[[VAND:.+]] = spirv.BitwiseAnd %[[VCAST]], %[[VVMASK]] : vector<3xi16>
//       CHECK:   %[[SAND:.+]] = spirv.BitwiseAnd %[[SCAST]], %[[SVMASK]] : vector<3xi16>
//       CHECK:   %[[OR:.+]] = spirv.BitwiseOr %[[VAND]], %[[SAND]] : vector<3xi16>
//       CHECK:   %[[RESULT:.+]] = spirv.Bitcast %[[OR]] : vector<3xi16> to vector<3xf16>
//       CHECK:   return %[[RESULT]]

// -----

// 2-D vectors are not supported.
func.func @copy_sign_2d_vector(%value: vector<3x3xf32>, %sign: vector<3x3xf32>) -> vector<3x3xf32> {
  %0 = math.copysign %value, %sign : vector<3x3xf32>
  return %0: vector<3x3xf32>
}

// CHECK-LABEL: func @copy_sign_2d_vector
// CHECK-NEXT:    math.copysign {{%.+}}, {{%.+}} : vector<3x3xf32>
// CHECK-NEXT:    return

// -----

// Tensors are not supported.
func.func @copy_sign_tensor(%value: tensor<3x3xf32>, %sign: tensor<3x3xf32>) -> tensor<3x3xf32> {
  %0 = math.copysign %value, %sign : tensor<3x3xf32>
  return %0: tensor<3x3xf32>
}

// CHECK-LABEL: func @copy_sign_tensor
// CHECK-NEXT:    math.copysign {{%.+}}, {{%.+}} : tensor<3x3xf32>
// CHECK-NEXT:    return
// -----

module attributes { spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Float16, Int16], []>, #spirv.resource_limits<>> } {

func.func @copy_sign_vector_0D(%value: vector<1xf16>, %sign: vector<1xf16>) -> vector<1xf16> {
  %0 = math.copysign %value, %sign : vector<1xf16>
  return %0: vector<1xf16>
}

}

// CHECK-LABEL: func @copy_sign_vector_0D
//  CHECK-SAME: (%[[VALUE:.+]]: vector<1xf16>, %[[SIGN:.+]]: vector<1xf16>)
//   CHECK-DAG:   %[[CASTVAL:.+]] = builtin.unrealized_conversion_cast %[[VALUE]] : vector<1xf16> to f16
//   CHECK-DAG:   %[[CASTSIGN:.+]] = builtin.unrealized_conversion_cast %[[SIGN]] : vector<1xf16> to f16
//       CHECK:   %[[SMASK:.+]] = spirv.Constant -32768 : i16
//       CHECK:   %[[VMASK:.+]] = spirv.Constant 32767 : i16
//       CHECK:   %[[VCAST:.+]] = spirv.Bitcast %[[CASTVAL]] : f16 to i16
//       CHECK:   %[[SCAST:.+]] = spirv.Bitcast %[[CASTSIGN]] : f16 to i16
//       CHECK:   %[[VAND:.+]] = spirv.BitwiseAnd %[[VCAST]], %[[VMASK]] : i16
//       CHECK:   %[[SAND:.+]] = spirv.BitwiseAnd %[[SCAST]], %[[SMASK]] : i16
//       CHECK:   %[[OR:.+]] = spirv.BitwiseOr %[[VAND]], %[[SAND]] : i16
//       CHECK:   %[[RESULT:.+]] = spirv.Bitcast %[[OR]] : i16 to f16
//       CHECK:   %[[CASTRESULT:.+]] = builtin.unrealized_conversion_cast %[[RESULT]] : f16 to vector<1xf16>
//       CHECK:   return %[[CASTRESULT]]
