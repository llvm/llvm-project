// RUN: mlir-opt -split-input-file -convert-to-spirv %s | FileCheck %s

func.func @create_complex(%real: f32, %imag: f32) -> complex<f32> {
  %0 = complex.create %real, %imag : complex<f32>
  return %0 : complex<f32>
}

// CHECK-LABEL: spirv.func @create_complex
//  CHECK-SAME: (%[[RE:.+]]: f32, %[[IM:.+]]: f32)
//       CHECK: %[[CC:.+]] = spirv.CompositeConstruct %[[RE]], %[[IM]] : (f32, f32) -> vector<2xf32>
//       CHECK: spirv.ReturnValue %[[CC]] : vector<2xf32>


// -----

func.func @real_number(%arg: complex<f32>) -> f32 {
  %real = complex.re %arg : complex<f32>
  return %real : f32
}

// CHECK-LABEL: spirv.func @real_number
//  CHECK-SAME: %[[ARG:.+]]: vector<2xf32>
//       CHECK: %[[RE:.+]] = spirv.CompositeExtract %[[ARG]][0 : i32] : vector<2xf32>
//       CHECK: spirv.ReturnValue %[[RE]] : f32

// -----

func.func @imaginary_number(%arg: complex<f32>) -> f32 {
  %imaginary = complex.im %arg : complex<f32>
  return %imaginary: f32
}

// CHECK-LABEL: spirv.func @imaginary_number
//  CHECK-SAME: %[[ARG:.+]]: vector<2xf32>
//       CHECK: %[[IM:.+]] = spirv.CompositeExtract %[[ARG]][1 : i32] : vector<2xf32>
//       CHECK: spirv.ReturnValue %[[IM]] : f32

// -----

func.func @complex_const() -> complex<f32> {
  %cst = complex.constant [0x7FC00000 : f32, 0.000000e+00 : f32] : complex<f32>
  return %cst : complex<f32>
}

// CHECK-LABEL: spirv.func @complex_const()
//       CHECK: spirv.Constant dense<[0x7FC00000, 0.000000e+00]> : vector<2xf32>
