// RUN: mlir-opt -split-input-file -convert-complex-to-spirv %s | FileCheck %s

func.func @create_complex(%real: f32, %imag: f32) -> complex<f32> {
  %0 = complex.create %real, %imag : complex<f32>
  return %0 : complex<f32>
}

// CHECK-LABEL: func.func @create_complex
//  CHECK-SAME: (%[[RE:.+]]: f32, %[[IM:.+]]: f32)
//       CHECK:   %[[CC:.+]] = spirv.CompositeConstruct %[[RE]], %[[IM]] : (f32, f32) -> vector<2xf32>
//       CHECK:   %[[CAST:.+]] = builtin.unrealized_conversion_cast %[[CC]] : vector<2xf32> to complex<f32>
//       CHECK:   return %[[CAST]] : complex<f32>


// -----

func.func @real_number(%arg: complex<f32>) -> f32 {
  %real = complex.re %arg : complex<f32>
  return %real : f32
}

// CHECK-LABEL: func.func @real_number
//  CHECK-SAME: %[[ARG:.+]]: complex<f32>
//       CHECK:   %[[CAST:.+]] = builtin.unrealized_conversion_cast %[[ARG]] : complex<f32> to vector<2xf32>
//       CHECK:   %[[RE:.+]] = spirv.CompositeExtract %[[CAST]][0 : i32] : vector<2xf32>
//       CHECK:   return %[[RE]] : f32

// -----

func.func @imaginary_number(%arg: complex<f32>) -> f32 {
  %imaginary = complex.im %arg : complex<f32>
  return %imaginary: f32
}

// CHECK-LABEL: func.func @imaginary_number
//  CHECK-SAME: %[[ARG:.+]]: complex<f32>
//       CHECK:   %[[CAST:.+]] = builtin.unrealized_conversion_cast %[[ARG]] : complex<f32> to vector<2xf32>
//       CHECK:   %[[IM:.+]] = spirv.CompositeExtract %[[CAST]][1 : i32] : vector<2xf32>
//       CHECK:   return %[[IM]] : f32

