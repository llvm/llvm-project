// RUN: mlir-opt -split-input-file -verify-diagnostics %s | FileCheck %s

//===----------------------------------------------------------------------===//
// spv.GLSL.Exp
//===----------------------------------------------------------------------===//

func @exp(%arg0 : f32) -> () {
  // CHECK: spv.GLSL.Exp {{%.*}} : f32
  %2 = spv.GLSL.Exp %arg0 : f32
  return
}

func @expvec(%arg0 : vector<3xf16>) -> () {
  // CHECK: spv.GLSL.Exp {{%.*}} : vector<3xf16>
  %2 = spv.GLSL.Exp %arg0 : vector<3xf16>
  return
}

// -----

func @exp(%arg0 : i32) -> () {
  // expected-error @+1 {{op operand #0 must be 16/32-bit float or vector of 16/32-bit float values}}
  %2 = spv.GLSL.Exp %arg0 : i32
  return
}

// -----

func @exp(%arg0 : vector<5xf32>) -> () {
  // expected-error @+1 {{op operand #0 must be 16/32-bit float or vector of 16/32-bit float values of length 2/3/4}}
  %2 = spv.GLSL.Exp %arg0 : vector<5xf32>
  return
}

// -----

func @exp(%arg0 : f32, %arg1 : f32) -> () {
  // expected-error @+1 {{expected ':'}}
  %2 = spv.GLSL.Exp %arg0, %arg1 : i32
  return
}

// -----

func @exp(%arg0 : i32) -> () {
  // expected-error @+2 {{expected non-function type}}
  %2 = spv.GLSL.Exp %arg0 :
  return
}

// -----

//===----------------------------------------------------------------------===//
// spv.GLSL.FMax
//===----------------------------------------------------------------------===//

func @fmax(%arg0 : f32, %arg1 : f32) -> () {
  // CHECK: spv.GLSL.FMax {{%.*}}, {{%.*}} : f32
  %2 = spv.GLSL.FMax %arg0, %arg1 : f32
  return
}

func @fmaxvec(%arg0 : vector<3xf16>, %arg1 : vector<3xf16>) -> () {
  // CHECK: spv.GLSL.FMax {{%.*}}, {{%.*}} : vector<3xf16>
  %2 = spv.GLSL.FMax %arg0, %arg1 : vector<3xf16>
  return
}

func @fmaxf64(%arg0 : f64, %arg1 : f64) -> () {
  // CHECK: spv.GLSL.FMax {{%.*}}, {{%.*}} : f64
  %2 = spv.GLSL.FMax %arg0, %arg1 : f64
  return
}

// -----

//===----------------------------------------------------------------------===//
// spv.GLSL.InverseSqrt
//===----------------------------------------------------------------------===//

func @inversesqrt(%arg0 : f32) -> () {
  // CHECK: spv.GLSL.InverseSqrt {{%.*}} : f32
  %2 = spv.GLSL.InverseSqrt %arg0 : f32
  return
}

func @inversesqrtvec(%arg0 : vector<3xf16>) -> () {
  // CHECK: spv.GLSL.InverseSqrt {{%.*}} : vector<3xf16>
  %2 = spv.GLSL.InverseSqrt %arg0 : vector<3xf16>
  return
}

// -----

//===----------------------------------------------------------------------===//
// spv.GLSL.Sqrt
//===----------------------------------------------------------------------===//

func @sqrt(%arg0 : f32) -> () {
  // CHECK: spv.GLSL.Sqrt {{%.*}} : f32
  %2 = spv.GLSL.Sqrt %arg0 : f32
  return
}

func @sqrtvec(%arg0 : vector<3xf16>) -> () {
  // CHECK: spv.GLSL.Sqrt {{%.*}} : vector<3xf16>
  %2 = spv.GLSL.Sqrt %arg0 : vector<3xf16>
  return
}

//===----------------------------------------------------------------------===//
// spv.GLSL.Cos
//===----------------------------------------------------------------------===//

func @cos(%arg0 : f32) -> () {
  // CHECK: spv.GLSL.Cos {{%.*}} : f32
  %2 = spv.GLSL.Cos %arg0 : f32
  return
}

func @cosvec(%arg0 : vector<3xf16>) -> () {
  // CHECK: spv.GLSL.Cos {{%.*}} : vector<3xf16>
  %2 = spv.GLSL.Cos %arg0 : vector<3xf16>
  return
}

//===----------------------------------------------------------------------===//
// spv.GLSL.Sin
//===----------------------------------------------------------------------===//

func @sin(%arg0 : f32) -> () {
  // CHECK: spv.GLSL.Sin {{%.*}} : f32
  %2 = spv.GLSL.Sin %arg0 : f32
  return
}

func @sinvec(%arg0 : vector<3xf16>) -> () {
  // CHECK: spv.GLSL.Sin {{%.*}} : vector<3xf16>
  %2 = spv.GLSL.Sin %arg0 : vector<3xf16>
  return
}

//===----------------------------------------------------------------------===//
// spv.GLSL.Tan
//===----------------------------------------------------------------------===//

func @tan(%arg0 : f32) -> () {
  // CHECK: spv.GLSL.Tan {{%.*}} : f32
  %2 = spv.GLSL.Tan %arg0 : f32
  return
}

func @tanvec(%arg0 : vector<3xf16>) -> () {
  // CHECK: spv.GLSL.Tan {{%.*}} : vector<3xf16>
  %2 = spv.GLSL.Tan %arg0 : vector<3xf16>
  return
}

//===----------------------------------------------------------------------===//
// spv.GLSL.Acos
//===----------------------------------------------------------------------===//

func @acos(%arg0 : f32) -> () {
  // CHECK: spv.GLSL.Acos {{%.*}} : f32
  %2 = spv.GLSL.Acos %arg0 : f32
  return
}

func @acosvec(%arg0 : vector<3xf16>) -> () {
  // CHECK: spv.GLSL.Acos {{%.*}} : vector<3xf16>
  %2 = spv.GLSL.Acos %arg0 : vector<3xf16>
  return
}

//===----------------------------------------------------------------------===//
// spv.GLSL.Asin
//===----------------------------------------------------------------------===//

func @asin(%arg0 : f32) -> () {
  // CHECK: spv.GLSL.Asin {{%.*}} : f32
  %2 = spv.GLSL.Asin %arg0 : f32
  return
}

func @asinvec(%arg0 : vector<3xf16>) -> () {
  // CHECK: spv.GLSL.Asin {{%.*}} : vector<3xf16>
  %2 = spv.GLSL.Asin %arg0 : vector<3xf16>
  return
}

//===----------------------------------------------------------------------===//
// spv.GLSL.Atan
//===----------------------------------------------------------------------===//

func @atan(%arg0 : f32) -> () {
  // CHECK: spv.GLSL.Atan {{%.*}} : f32
  %2 = spv.GLSL.Atan %arg0 : f32
  return
}

func @atanvec(%arg0 : vector<3xf16>) -> () {
  // CHECK: spv.GLSL.Atan {{%.*}} : vector<3xf16>
  %2 = spv.GLSL.Atan %arg0 : vector<3xf16>
  return
}

//===----------------------------------------------------------------------===//
// spv.GLSL.Sinh
//===----------------------------------------------------------------------===//

func @sinh(%arg0 : f32) -> () {
  // CHECK: spv.GLSL.Sinh {{%.*}} : f32
  %2 = spv.GLSL.Sinh %arg0 : f32
  return
}

func @sinhvec(%arg0 : vector<3xf16>) -> () {
  // CHECK: spv.GLSL.Sinh {{%.*}} : vector<3xf16>
  %2 = spv.GLSL.Sinh %arg0 : vector<3xf16>
  return
}

//===----------------------------------------------------------------------===//
// spv.GLSL.Cosh
//===----------------------------------------------------------------------===//

func @cosh(%arg0 : f32) -> () {
  // CHECK: spv.GLSL.Cosh {{%.*}} : f32
  %2 = spv.GLSL.Cosh %arg0 : f32
  return
}

func @coshvec(%arg0 : vector<3xf16>) -> () {
  // CHECK: spv.GLSL.Cosh {{%.*}} : vector<3xf16>
  %2 = spv.GLSL.Cosh %arg0 : vector<3xf16>
  return
}

//===----------------------------------------------------------------------===//
// spv.GLSL.Pow
//===----------------------------------------------------------------------===//

func @pow(%arg0 : f32, %arg1 : f32) -> () {
  // CHECK: spv.GLSL.Pow {{%.*}}, {{%.*}} : f32
  %2 = spv.GLSL.Pow %arg0, %arg1 : f32
  return
}

func @powvec(%arg0 : vector<3xf16>, %arg1 : vector<3xf16>) -> () {
  // CHECK: spv.GLSL.Pow {{%.*}}, {{%.*}} : vector<3xf16>
  %2 = spv.GLSL.Pow %arg0, %arg1 : vector<3xf16>
  return
}

// -----

//===----------------------------------------------------------------------===//
// spv.GLSL.Round
//===----------------------------------------------------------------------===//

func @round(%arg0 : f32) -> () {
  // CHECK: spv.GLSL.Round {{%.*}} : f32
  %2 = spv.GLSL.Round %arg0 : f32
  return
}

func @roundvec(%arg0 : vector<3xf16>) -> () {
  // CHECK: spv.GLSL.Round {{%.*}} : vector<3xf16>
  %2 = spv.GLSL.Round %arg0 : vector<3xf16>
  return
}

// -----

//===----------------------------------------------------------------------===//
// spv.GLSL.FClamp
//===----------------------------------------------------------------------===//

func @fclamp(%arg0 : f32, %min : f32, %max : f32) -> () {
  // CHECK: spv.GLSL.FClamp {{%[^,]*}}, {{%[^,]*}}, {{%[^,]*}} : f32
  %2 = spv.GLSL.FClamp %arg0, %min, %max : f32
  return
}

// -----

func @fclamp(%arg0 : vector<3xf32>, %min : vector<3xf32>, %max : vector<3xf32>) -> () {
  // CHECK: spv.GLSL.FClamp {{%[^,]*}}, {{%[^,]*}}, {{%[^,]*}} : vector<3xf32>
  %2 = spv.GLSL.FClamp %arg0, %min, %max : vector<3xf32>
  return
}

// -----

//===----------------------------------------------------------------------===//
// spv.GLSL.UClamp
//===----------------------------------------------------------------------===//

func @fclamp(%arg0 : ui32, %min : ui32, %max : ui32) -> () {
  // CHECK: spv.GLSL.UClamp {{%[^,]*}}, {{%[^,]*}}, {{%[^,]*}} : ui32
  %2 = spv.GLSL.UClamp %arg0, %min, %max : ui32
  return
}

// -----

func @fclamp(%arg0 : vector<4xi32>, %min : vector<4xi32>, %max : vector<4xi32>) -> () {
  // CHECK: spv.GLSL.UClamp {{%[^,]*}}, {{%[^,]*}}, {{%[^,]*}} : vector<4xi32>
  %2 = spv.GLSL.UClamp %arg0, %min, %max : vector<4xi32>
  return
}

// -----

func @fclamp(%arg0 : si32, %min : si32, %max : si32) -> () {
  // expected-error @+1 {{must be 8/16/32/64-bit signless/unsigned integer or vector}}
  %2 = spv.GLSL.UClamp %arg0, %min, %max : si32
  return
}

// -----

//===----------------------------------------------------------------------===//
// spv.GLSL.SClamp
//===----------------------------------------------------------------------===//

func @fclamp(%arg0 : si32, %min : si32, %max : si32) -> () {
  // CHECK: spv.GLSL.SClamp {{%[^,]*}}, {{%[^,]*}}, {{%[^,]*}} : si32
  %2 = spv.GLSL.SClamp %arg0, %min, %max : si32
  return
}

// -----

func @fclamp(%arg0 : vector<4xsi32>, %min : vector<4xsi32>, %max : vector<4xsi32>) -> () {
  // CHECK: spv.GLSL.SClamp {{%[^,]*}}, {{%[^,]*}}, {{%[^,]*}} : vector<4xsi32>
  %2 = spv.GLSL.SClamp %arg0, %min, %max : vector<4xsi32>
  return
}

// -----

func @fclamp(%arg0 : i32, %min : i32, %max : i32) -> () {
  // expected-error @+1 {{must be 8/16/32/64-bit signed integer or vector}}
  %2 = spv.GLSL.SClamp %arg0, %min, %max : i32
  return
}

// -----

//===----------------------------------------------------------------------===//
// spv.GLSL.Fma
//===----------------------------------------------------------------------===//

func @fma(%a : f32, %b : f32, %c : f32) -> () {
  // CHECK: spv.GLSL.Fma {{%[^,]*}}, {{%[^,]*}}, {{%[^,]*}} : f32
  %2 = spv.GLSL.Fma %a, %b, %c : f32
  return
}

// -----

func @fma(%a : vector<3xf32>, %b : vector<3xf32>, %c : vector<3xf32>) -> () {
  // CHECK: spv.GLSL.Fma {{%[^,]*}}, {{%[^,]*}}, {{%[^,]*}} : vector<3xf32>
  %2 = spv.GLSL.Fma %a, %b, %c : vector<3xf32>
  return
}
// -----

//===----------------------------------------------------------------------===//
// spv.GLSL.FrexpStruct
//===----------------------------------------------------------------------===//

func @frexp_struct(%arg0 : f32) -> () {
  // CHECK: spv.GLSL.FrexpStruct {{%.*}} : f32 -> !spv.struct<(f32, i32)>
  %2 = spv.GLSL.FrexpStruct %arg0 : f32 -> !spv.struct<(f32, i32)>
  return
}

func @frexp_struct_64(%arg0 : f64) -> () {
  // CHECK: spv.GLSL.FrexpStruct {{%.*}} : f64 -> !spv.struct<(f64, i32)>
  %2 = spv.GLSL.FrexpStruct %arg0 : f64 -> !spv.struct<(f64, i32)>
  return
}

func @frexp_struct_vec(%arg0 : vector<3xf32>) -> () {
  // CHECK: spv.GLSL.FrexpStruct {{%.*}} : vector<3xf32> -> !spv.struct<(vector<3xf32>, vector<3xi32>)>
  %2 = spv.GLSL.FrexpStruct %arg0 : vector<3xf32> -> !spv.struct<(vector<3xf32>, vector<3xi32>)>
  return
}

// -----

func @frexp_struct_mismatch_type(%arg0 : f32) -> () {
  // expected-error @+1 {{member zero of the resulting struct type must be the same type as the operand}}
  %2 = spv.GLSL.FrexpStruct %arg0 : f32 -> !spv.struct<(vector<3xf32>, i32)>
  return
}

// -----

func @frexp_struct_wrong_type(%arg0 : i32) -> () {
  // expected-error @+1 {{op operand #0 must be 16/32/64-bit float or vector of 16/32/64-bit float values}}
  %2 = spv.GLSL.FrexpStruct %arg0 : i32 -> !spv.struct<(i32, i32)>
  return
}

// -----

func @frexp_struct_mismatch_num_components(%arg0 : vector<3xf32>) -> () {
  // expected-error @+1 {{member one of the resulting struct type must have the same number of components as the operand type}}
  %2 = spv.GLSL.FrexpStruct %arg0 : vector<3xf32> -> !spv.struct<(vector<3xf32>, vector<2xi32>)>
  return
}

// -----

func @frexp_struct_not_i32(%arg0 : f32) -> () {
  // expected-error @+1 {{member one of the resulting struct type must be a scalar or vector of 32 bit integer type}}
  %2 = spv.GLSL.FrexpStruct %arg0 : f32 -> !spv.struct<(f32, i64)>
  return
}
