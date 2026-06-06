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

// -----

func.func @complex_const() -> complex<f32> {
  %cst = complex.constant [0x7FC00000 : f32, 0.000000e+00 : f32] : complex<f32>
  return %cst : complex<f32>
}

// CHECK-LABEL: func.func @complex_const()
//       CHECK:   spirv.Constant dense<[0x7FC00000, 0.000000e+00]> : vector<2xf32>

// -----

func.func @complex_add(%lhs: complex<f32>, %rhs: complex<f32>) -> complex<f32> {
  %0 = complex.add %lhs, %rhs : complex<f32>
  return %0 : complex<f32>
}

// CHECK-LABEL: func.func @complex_add
//  CHECK-SAME: (%[[LHS:.+]]: complex<f32>, %[[RHS:.+]]: complex<f32>)
//   CHECK-DAG:   %[[LV:.+]] = builtin.unrealized_conversion_cast %[[LHS]] : complex<f32> to vector<2xf32>
//   CHECK-DAG:   %[[RV:.+]] = builtin.unrealized_conversion_cast %[[RHS]] : complex<f32> to vector<2xf32>
//       CHECK:   %[[LRE:.+]] = spirv.CompositeExtract %[[LV]][0 : i32] : vector<2xf32>
//       CHECK:   %[[LIM:.+]] = spirv.CompositeExtract %[[LV]][1 : i32] : vector<2xf32>
//       CHECK:   %[[RRE:.+]] = spirv.CompositeExtract %[[RV]][0 : i32] : vector<2xf32>
//       CHECK:   %[[RIM:.+]] = spirv.CompositeExtract %[[RV]][1 : i32] : vector<2xf32>
//       CHECK:   %[[RE:.+]] = spirv.FAdd %[[LRE]], %[[RRE]] : f32
//       CHECK:   %[[IM:.+]] = spirv.FAdd %[[LIM]], %[[RIM]] : f32
//       CHECK:   %[[CC:.+]] = spirv.CompositeConstruct %[[RE]], %[[IM]] : (f32, f32) -> vector<2xf32>

// -----

func.func @complex_sub(%lhs: complex<f32>, %rhs: complex<f32>) -> complex<f32> {
  %0 = complex.sub %lhs, %rhs : complex<f32>
  return %0 : complex<f32>
}

// CHECK-LABEL: func.func @complex_sub
//       CHECK:   spirv.FSub
//       CHECK:   spirv.FSub
//       CHECK:   spirv.CompositeConstruct

// -----

func.func @complex_mul(%lhs: complex<f32>, %rhs: complex<f32>) -> complex<f32> {
  %0 = complex.mul %lhs, %rhs : complex<f32>
  return %0 : complex<f32>
}

// CHECK-LABEL: func.func @complex_mul
//  CHECK-SAME: (%[[LHS:.+]]: complex<f32>, %[[RHS:.+]]: complex<f32>)
//   CHECK-DAG:   %[[LV:.+]] = builtin.unrealized_conversion_cast %[[LHS]] : complex<f32> to vector<2xf32>
//   CHECK-DAG:   %[[RV:.+]] = builtin.unrealized_conversion_cast %[[RHS]] : complex<f32> to vector<2xf32>
//       CHECK:   %[[A:.+]] = spirv.CompositeExtract %[[LV]][0 : i32] : vector<2xf32>
//       CHECK:   %[[B:.+]] = spirv.CompositeExtract %[[LV]][1 : i32] : vector<2xf32>
//       CHECK:   %[[C:.+]] = spirv.CompositeExtract %[[RV]][0 : i32] : vector<2xf32>
//       CHECK:   %[[D:.+]] = spirv.CompositeExtract %[[RV]][1 : i32] : vector<2xf32>
//       CHECK:   %[[AC:.+]] = spirv.FMul %[[A]], %[[C]] : f32
//       CHECK:   %[[BD:.+]] = spirv.FMul %[[B]], %[[D]] : f32
//       CHECK:   %[[AD:.+]] = spirv.FMul %[[A]], %[[D]] : f32
//       CHECK:   %[[BC:.+]] = spirv.FMul %[[B]], %[[C]] : f32
//       CHECK:   %[[RE:.+]] = spirv.FSub %[[AC]], %[[BD]] : f32
//       CHECK:   %[[IM:.+]] = spirv.FAdd %[[AD]], %[[BC]] : f32
//       CHECK:   %[[CC:.+]] = spirv.CompositeConstruct %[[RE]], %[[IM]] : (f32, f32) -> vector<2xf32>

// -----

func.func @complex_div(%lhs: complex<f32>, %rhs: complex<f32>) -> complex<f32> {
  %0 = complex.div %lhs, %rhs : complex<f32>
  return %0 : complex<f32>
}

// CHECK-LABEL: func.func @complex_div
//  CHECK-SAME: (%[[LHS:.+]]: complex<f32>, %[[RHS:.+]]: complex<f32>)
//   CHECK-DAG:   %[[LV:.+]] = builtin.unrealized_conversion_cast %[[LHS]] : complex<f32> to vector<2xf32>
//   CHECK-DAG:   %[[RV:.+]] = builtin.unrealized_conversion_cast %[[RHS]] : complex<f32> to vector<2xf32>
//       CHECK:   %[[A:.+]] = spirv.CompositeExtract %[[LV]][0 : i32] : vector<2xf32>
//       CHECK:   %[[B:.+]] = spirv.CompositeExtract %[[LV]][1 : i32] : vector<2xf32>
//       CHECK:   %[[C:.+]] = spirv.CompositeExtract %[[RV]][0 : i32] : vector<2xf32>
//       CHECK:   %[[D:.+]] = spirv.CompositeExtract %[[RV]][1 : i32] : vector<2xf32>
//       CHECK:   %[[AC:.+]] = spirv.FMul %[[A]], %[[C]] : f32
//       CHECK:   %[[BD:.+]] = spirv.FMul %[[B]], %[[D]] : f32
//       CHECK:   %[[BC:.+]] = spirv.FMul %[[B]], %[[C]] : f32
//       CHECK:   %[[AD:.+]] = spirv.FMul %[[A]], %[[D]] : f32
//       CHECK:   %[[CC2:.+]] = spirv.FMul %[[C]], %[[C]] : f32
//       CHECK:   %[[DD:.+]] = spirv.FMul %[[D]], %[[D]] : f32
//       CHECK:   %[[DENOM:.+]] = spirv.FAdd %[[CC2]], %[[DD]] : f32
//       CHECK:   %[[NRE:.+]] = spirv.FAdd %[[AC]], %[[BD]] : f32
//       CHECK:   %[[NIM:.+]] = spirv.FSub %[[BC]], %[[AD]] : f32
//       CHECK:   %[[RE:.+]] = spirv.FDiv %[[NRE]], %[[DENOM]] : f32
//       CHECK:   %[[IM:.+]] = spirv.FDiv %[[NIM]], %[[DENOM]] : f32
//       CHECK:   %[[CC:.+]] = spirv.CompositeConstruct %[[RE]], %[[IM]] : (f32, f32) -> vector<2xf32>

// -----

func.func @complex_abs(%arg: complex<f32>) -> f32 {
  %abs = complex.abs %arg : complex<f32>
  return %abs : f32
}

// CHECK-LABEL: func.func @complex_abs
//  CHECK-SAME: %[[ARG:.+]]: complex<f32>
//       CHECK:   %[[V:.+]] = builtin.unrealized_conversion_cast %[[ARG]] : complex<f32> to vector<2xf32>
//       CHECK:   %[[RE:.+]] = spirv.CompositeExtract %[[V]][0 : i32] : vector<2xf32>
//       CHECK:   %[[IM:.+]] = spirv.CompositeExtract %[[V]][1 : i32] : vector<2xf32>
//       CHECK:   %[[RESQ:.+]] = spirv.FMul %[[RE]], %[[RE]] : f32
//       CHECK:   %[[IMSQ:.+]] = spirv.FMul %[[IM]], %[[IM]] : f32
//       CHECK:   %[[SUM:.+]] = spirv.FAdd %[[RESQ]], %[[IMSQ]] : f32
//       CHECK:   %[[ABS:.+]] = spirv.GL.Sqrt %[[SUM]] : f32
//       CHECK:   return %[[ABS]] : f32

// -----

module attributes {
  spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Kernel], []>, #spirv.resource_limits<>>
} {

func.func @complex_abs_opencl(%arg: complex<f32>) -> f32 {
  %abs = complex.abs %arg : complex<f32>
  return %abs : f32
}

// CHECK-LABEL: func.func @complex_abs_opencl
//       CHECK:   spirv.FMul
//       CHECK:   spirv.FMul
//       CHECK:   %[[SUM:.+]] = spirv.FAdd
//       CHECK:   spirv.CL.sqrt %[[SUM]] : f32

}
