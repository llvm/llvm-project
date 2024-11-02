// RUN: mlir-opt -split-input-file -convert-math-to-spirv -verify-diagnostics %s -o - | FileCheck %s

func.func @copy_sign_scalar(%value: f32, %sign: f32) -> f32 {
  %0 = math.copysign %value, %sign : f32
  return %0: f32
}

// CHECK-LABEL: func @copy_sign_scalar
//  CHECK-SAME: (%[[VALUE:.+]]: f32, %[[SIGN:.+]]: f32)
//       CHECK:   %[[SMASK:.+]] = spv.Constant -2147483648 : i32
//       CHECK:   %[[VMASK:.+]] = spv.Constant 2147483647 : i32
//       CHECK:   %[[VCAST:.+]] = spv.Bitcast %[[VALUE]] : f32 to i32
//       CHECK:   %[[SCAST:.+]] = spv.Bitcast %[[SIGN]] : f32 to i32
//       CHECK:   %[[VAND:.+]] = spv.BitwiseAnd %[[VCAST]], %[[VMASK]] : i32
//       CHECK:   %[[SAND:.+]] = spv.BitwiseAnd %[[SCAST]], %[[SMASK]] : i32
//       CHECK:   %[[OR:.+]] = spv.BitwiseOr %[[VAND]], %[[SAND]] : i32
//       CHECK:   %[[RESULT:.+]] = spv.Bitcast %[[OR]] : i32 to f32
//       CHECK:   return %[[RESULT]]

// -----

module attributes { spv.target_env = #spv.target_env<#spv.vce<v1.0, [Float16, Int16], []>, {}> } {

func.func @copy_sign_vector(%value: vector<3xf16>, %sign: vector<3xf16>) -> vector<3xf16> {
  %0 = math.copysign %value, %sign : vector<3xf16>
  return %0: vector<3xf16>
}

}

// CHECK-LABEL: func @copy_sign_vector
//  CHECK-SAME: (%[[VALUE:.+]]: vector<3xf16>, %[[SIGN:.+]]: vector<3xf16>)
//       CHECK:   %[[SMASK:.+]] = spv.Constant -32768 : i16
//       CHECK:   %[[VMASK:.+]] = spv.Constant 32767 : i16
//       CHECK:   %[[SVMASK:.+]] = spv.CompositeConstruct %[[SMASK]], %[[SMASK]], %[[SMASK]] : vector<3xi16>
//       CHECK:   %[[VVMASK:.+]] = spv.CompositeConstruct %[[VMASK]], %[[VMASK]], %[[VMASK]] : vector<3xi16>
//       CHECK:   %[[VCAST:.+]] = spv.Bitcast %[[VALUE]] : vector<3xf16> to vector<3xi16>
//       CHECK:   %[[SCAST:.+]] = spv.Bitcast %[[SIGN]] : vector<3xf16> to vector<3xi16>
//       CHECK:   %[[VAND:.+]] = spv.BitwiseAnd %[[VCAST]], %[[VVMASK]] : vector<3xi16>
//       CHECK:   %[[SAND:.+]] = spv.BitwiseAnd %[[SCAST]], %[[SVMASK]] : vector<3xi16>
//       CHECK:   %[[OR:.+]] = spv.BitwiseOr %[[VAND]], %[[SAND]] : vector<3xi16>
//       CHECK:   %[[RESULT:.+]] = spv.Bitcast %[[OR]] : vector<3xi16> to vector<3xf16>
//       CHECK:   return %[[RESULT]]
