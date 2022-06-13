// RUN: mlir-opt -split-input-file -verify-diagnostics %s | FileCheck %s

//===----------------------------------------------------------------------===//
// spv.GLSL.Exp
//===----------------------------------------------------------------------===//

func.func @exp(%arg0 : f32) -> () {
  // CHECK: spv.GLSL.Exp {{%.*}} : f32
  %2 = spv.GLSL.Exp %arg0 : f32
  return
}

func.func @expvec(%arg0 : vector<3xf16>) -> () {
  // CHECK: spv.GLSL.Exp {{%.*}} : vector<3xf16>
  %2 = spv.GLSL.Exp %arg0 : vector<3xf16>
  return
}

// -----

func.func @exp(%arg0 : i32) -> () {
  // expected-error @+1 {{op operand #0 must be 16/32-bit float or vector of 16/32-bit float values}}
  %2 = spv.GLSL.Exp %arg0 : i32
  return
}

// -----

func.func @exp(%arg0 : vector<5xf32>) -> () {
  // expected-error @+1 {{op operand #0 must be 16/32-bit float or vector of 16/32-bit float values of length 2/3/4}}
  %2 = spv.GLSL.Exp %arg0 : vector<5xf32>
  return
}

// -----

func.func @exp(%arg0 : f32, %arg1 : f32) -> () {
  // expected-error @+1 {{expected ':'}}
  %2 = spv.GLSL.Exp %arg0, %arg1 : i32
  return
}

// -----

func.func @exp(%arg0 : i32) -> () {
  // expected-error @+1 {{expected non-function type}}
  %2 = spv.GLSL.Exp %arg0 :
  return
}

// -----

//===----------------------------------------------------------------------===//
// spv.GLSL.{F|S|U}{Max|Min}
//===----------------------------------------------------------------------===//

func.func @fmaxmin(%arg0 : f32, %arg1 : f32) {
  // CHECK: spv.GLSL.FMax {{%.*}}, {{%.*}} : f32
  %1 = spv.GLSL.FMax %arg0, %arg1 : f32
  // CHECK: spv.GLSL.FMin {{%.*}}, {{%.*}} : f32
  %2 = spv.GLSL.FMin %arg0, %arg1 : f32
  return
}

func.func @fmaxminvec(%arg0 : vector<3xf16>, %arg1 : vector<3xf16>) {
  // CHECK: spv.GLSL.FMax {{%.*}}, {{%.*}} : vector<3xf16>
  %1 = spv.GLSL.FMax %arg0, %arg1 : vector<3xf16>
  // CHECK: spv.GLSL.FMin {{%.*}}, {{%.*}} : vector<3xf16>
  %2 = spv.GLSL.FMin %arg0, %arg1 : vector<3xf16>
  return
}

func.func @fmaxminf64(%arg0 : f64, %arg1 : f64) {
  // CHECK: spv.GLSL.FMax {{%.*}}, {{%.*}} : f64
  %1 = spv.GLSL.FMax %arg0, %arg1 : f64
  // CHECK: spv.GLSL.FMin {{%.*}}, {{%.*}} : f64
  %2 = spv.GLSL.FMin %arg0, %arg1 : f64
  return
}

func.func @iminmax(%arg0: i32, %arg1: i32) {
  // CHECK: spv.GLSL.SMax {{%.*}}, {{%.*}} : i32
  %1 = spv.GLSL.SMax %arg0, %arg1 : i32
  // CHECK: spv.GLSL.UMax {{%.*}}, {{%.*}} : i32
  %2 = spv.GLSL.UMax %arg0, %arg1 : i32
  // CHECK: spv.GLSL.SMin {{%.*}}, {{%.*}} : i32
  %3 = spv.GLSL.SMin %arg0, %arg1 : i32
  // CHECK: spv.GLSL.UMin {{%.*}}, {{%.*}} : i32
  %4 = spv.GLSL.UMin %arg0, %arg1 : i32
  return
}

// -----

//===----------------------------------------------------------------------===//
// spv.GLSL.InverseSqrt
//===----------------------------------------------------------------------===//

func.func @inversesqrt(%arg0 : f32) -> () {
  // CHECK: spv.GLSL.InverseSqrt {{%.*}} : f32
  %2 = spv.GLSL.InverseSqrt %arg0 : f32
  return
}

func.func @inversesqrtvec(%arg0 : vector<3xf16>) -> () {
  // CHECK: spv.GLSL.InverseSqrt {{%.*}} : vector<3xf16>
  %2 = spv.GLSL.InverseSqrt %arg0 : vector<3xf16>
  return
}

// -----

//===----------------------------------------------------------------------===//
// spv.GLSL.Sqrt
//===----------------------------------------------------------------------===//

func.func @sqrt(%arg0 : f32) -> () {
  // CHECK: spv.GLSL.Sqrt {{%.*}} : f32
  %2 = spv.GLSL.Sqrt %arg0 : f32
  return
}

func.func @sqrtvec(%arg0 : vector<3xf16>) -> () {
  // CHECK: spv.GLSL.Sqrt {{%.*}} : vector<3xf16>
  %2 = spv.GLSL.Sqrt %arg0 : vector<3xf16>
  return
}

//===----------------------------------------------------------------------===//
// spv.GLSL.Cos
//===----------------------------------------------------------------------===//

func.func @cos(%arg0 : f32) -> () {
  // CHECK: spv.GLSL.Cos {{%.*}} : f32
  %2 = spv.GLSL.Cos %arg0 : f32
  return
}

func.func @cosvec(%arg0 : vector<3xf16>) -> () {
  // CHECK: spv.GLSL.Cos {{%.*}} : vector<3xf16>
  %2 = spv.GLSL.Cos %arg0 : vector<3xf16>
  return
}

//===----------------------------------------------------------------------===//
// spv.GLSL.Sin
//===----------------------------------------------------------------------===//

func.func @sin(%arg0 : f32) -> () {
  // CHECK: spv.GLSL.Sin {{%.*}} : f32
  %2 = spv.GLSL.Sin %arg0 : f32
  return
}

func.func @sinvec(%arg0 : vector<3xf16>) -> () {
  // CHECK: spv.GLSL.Sin {{%.*}} : vector<3xf16>
  %2 = spv.GLSL.Sin %arg0 : vector<3xf16>
  return
}

//===----------------------------------------------------------------------===//
// spv.GLSL.Tan
//===----------------------------------------------------------------------===//

func.func @tan(%arg0 : f32) -> () {
  // CHECK: spv.GLSL.Tan {{%.*}} : f32
  %2 = spv.GLSL.Tan %arg0 : f32
  return
}

func.func @tanvec(%arg0 : vector<3xf16>) -> () {
  // CHECK: spv.GLSL.Tan {{%.*}} : vector<3xf16>
  %2 = spv.GLSL.Tan %arg0 : vector<3xf16>
  return
}

//===----------------------------------------------------------------------===//
// spv.GLSL.Acos
//===----------------------------------------------------------------------===//

func.func @acos(%arg0 : f32) -> () {
  // CHECK: spv.GLSL.Acos {{%.*}} : f32
  %2 = spv.GLSL.Acos %arg0 : f32
  return
}

func.func @acosvec(%arg0 : vector<3xf16>) -> () {
  // CHECK: spv.GLSL.Acos {{%.*}} : vector<3xf16>
  %2 = spv.GLSL.Acos %arg0 : vector<3xf16>
  return
}

//===----------------------------------------------------------------------===//
// spv.GLSL.Asin
//===----------------------------------------------------------------------===//

func.func @asin(%arg0 : f32) -> () {
  // CHECK: spv.GLSL.Asin {{%.*}} : f32
  %2 = spv.GLSL.Asin %arg0 : f32
  return
}

func.func @asinvec(%arg0 : vector<3xf16>) -> () {
  // CHECK: spv.GLSL.Asin {{%.*}} : vector<3xf16>
  %2 = spv.GLSL.Asin %arg0 : vector<3xf16>
  return
}

//===----------------------------------------------------------------------===//
// spv.GLSL.Atan
//===----------------------------------------------------------------------===//

func.func @atan(%arg0 : f32) -> () {
  // CHECK: spv.GLSL.Atan {{%.*}} : f32
  %2 = spv.GLSL.Atan %arg0 : f32
  return
}

func.func @atanvec(%arg0 : vector<3xf16>) -> () {
  // CHECK: spv.GLSL.Atan {{%.*}} : vector<3xf16>
  %2 = spv.GLSL.Atan %arg0 : vector<3xf16>
  return
}

//===----------------------------------------------------------------------===//
// spv.GLSL.Sinh
//===----------------------------------------------------------------------===//

func.func @sinh(%arg0 : f32) -> () {
  // CHECK: spv.GLSL.Sinh {{%.*}} : f32
  %2 = spv.GLSL.Sinh %arg0 : f32
  return
}

func.func @sinhvec(%arg0 : vector<3xf16>) -> () {
  // CHECK: spv.GLSL.Sinh {{%.*}} : vector<3xf16>
  %2 = spv.GLSL.Sinh %arg0 : vector<3xf16>
  return
}

//===----------------------------------------------------------------------===//
// spv.GLSL.Cosh
//===----------------------------------------------------------------------===//

func.func @cosh(%arg0 : f32) -> () {
  // CHECK: spv.GLSL.Cosh {{%.*}} : f32
  %2 = spv.GLSL.Cosh %arg0 : f32
  return
}

func.func @coshvec(%arg0 : vector<3xf16>) -> () {
  // CHECK: spv.GLSL.Cosh {{%.*}} : vector<3xf16>
  %2 = spv.GLSL.Cosh %arg0 : vector<3xf16>
  return
}

//===----------------------------------------------------------------------===//
// spv.GLSL.Pow
//===----------------------------------------------------------------------===//

func.func @pow(%arg0 : f32, %arg1 : f32) -> () {
  // CHECK: spv.GLSL.Pow {{%.*}}, {{%.*}} : f32
  %2 = spv.GLSL.Pow %arg0, %arg1 : f32
  return
}

func.func @powvec(%arg0 : vector<3xf16>, %arg1 : vector<3xf16>) -> () {
  // CHECK: spv.GLSL.Pow {{%.*}}, {{%.*}} : vector<3xf16>
  %2 = spv.GLSL.Pow %arg0, %arg1 : vector<3xf16>
  return
}

// -----

//===----------------------------------------------------------------------===//
// spv.GLSL.Round
//===----------------------------------------------------------------------===//

func.func @round(%arg0 : f32) -> () {
  // CHECK: spv.GLSL.Round {{%.*}} : f32
  %2 = spv.GLSL.Round %arg0 : f32
  return
}

func.func @roundvec(%arg0 : vector<3xf16>) -> () {
  // CHECK: spv.GLSL.Round {{%.*}} : vector<3xf16>
  %2 = spv.GLSL.Round %arg0 : vector<3xf16>
  return
}

// -----

//===----------------------------------------------------------------------===//
// spv.GLSL.FClamp
//===----------------------------------------------------------------------===//

func.func @fclamp(%arg0 : f32, %min : f32, %max : f32) -> () {
  // CHECK: spv.GLSL.FClamp {{%[^,]*}}, {{%[^,]*}}, {{%[^,]*}} : f32
  %2 = spv.GLSL.FClamp %arg0, %min, %max : f32
  return
}

// -----

func.func @fclamp(%arg0 : vector<3xf32>, %min : vector<3xf32>, %max : vector<3xf32>) -> () {
  // CHECK: spv.GLSL.FClamp {{%[^,]*}}, {{%[^,]*}}, {{%[^,]*}} : vector<3xf32>
  %2 = spv.GLSL.FClamp %arg0, %min, %max : vector<3xf32>
  return
}

// -----

//===----------------------------------------------------------------------===//
// spv.GLSL.UClamp
//===----------------------------------------------------------------------===//

func.func @uclamp(%arg0 : ui32, %min : ui32, %max : ui32) -> () {
  // CHECK: spv.GLSL.UClamp {{%[^,]*}}, {{%[^,]*}}, {{%[^,]*}} : ui32
  %2 = spv.GLSL.UClamp %arg0, %min, %max : ui32
  return
}

// -----

func.func @uclamp(%arg0 : vector<4xi32>, %min : vector<4xi32>, %max : vector<4xi32>) -> () {
  // CHECK: spv.GLSL.UClamp {{%[^,]*}}, {{%[^,]*}}, {{%[^,]*}} : vector<4xi32>
  %2 = spv.GLSL.UClamp %arg0, %min, %max : vector<4xi32>
  return
}

// -----

func.func @uclamp(%arg0 : si32, %min : si32, %max : si32) -> () {
  // CHECK: spv.GLSL.UClamp
  %2 = spv.GLSL.UClamp %arg0, %min, %max : si32
  return
}

// -----

//===----------------------------------------------------------------------===//
// spv.GLSL.SClamp
//===----------------------------------------------------------------------===//

func.func @sclamp(%arg0 : si32, %min : si32, %max : si32) -> () {
  // CHECK: spv.GLSL.SClamp {{%[^,]*}}, {{%[^,]*}}, {{%[^,]*}} : si32
  %2 = spv.GLSL.SClamp %arg0, %min, %max : si32
  return
}

// -----

func.func @sclamp(%arg0 : vector<4xsi32>, %min : vector<4xsi32>, %max : vector<4xsi32>) -> () {
  // CHECK: spv.GLSL.SClamp {{%[^,]*}}, {{%[^,]*}}, {{%[^,]*}} : vector<4xsi32>
  %2 = spv.GLSL.SClamp %arg0, %min, %max : vector<4xsi32>
  return
}

// -----

func.func @sclamp(%arg0 : i32, %min : i32, %max : i32) -> () {
  // CHECK: spv.GLSL.SClamp
  %2 = spv.GLSL.SClamp %arg0, %min, %max : i32
  return
}

// -----

//===----------------------------------------------------------------------===//
// spv.GLSL.Fma
//===----------------------------------------------------------------------===//

func.func @fma(%a : f32, %b : f32, %c : f32) -> () {
  // CHECK: spv.GLSL.Fma {{%[^,]*}}, {{%[^,]*}}, {{%[^,]*}} : f32
  %2 = spv.GLSL.Fma %a, %b, %c : f32
  return
}

// -----

func.func @fma(%a : vector<3xf32>, %b : vector<3xf32>, %c : vector<3xf32>) -> () {
  // CHECK: spv.GLSL.Fma {{%[^,]*}}, {{%[^,]*}}, {{%[^,]*}} : vector<3xf32>
  %2 = spv.GLSL.Fma %a, %b, %c : vector<3xf32>
  return
}
// -----

//===----------------------------------------------------------------------===//
// spv.GLSL.FrexpStruct
//===----------------------------------------------------------------------===//

func.func @frexp_struct(%arg0 : f32) -> () {
  // CHECK: spv.GLSL.FrexpStruct {{%.*}} : f32 -> !spv.struct<(f32, i32)>
  %2 = spv.GLSL.FrexpStruct %arg0 : f32 -> !spv.struct<(f32, i32)>
  return
}

func.func @frexp_struct_64(%arg0 : f64) -> () {
  // CHECK: spv.GLSL.FrexpStruct {{%.*}} : f64 -> !spv.struct<(f64, i32)>
  %2 = spv.GLSL.FrexpStruct %arg0 : f64 -> !spv.struct<(f64, i32)>
  return
}

func.func @frexp_struct_vec(%arg0 : vector<3xf32>) -> () {
  // CHECK: spv.GLSL.FrexpStruct {{%.*}} : vector<3xf32> -> !spv.struct<(vector<3xf32>, vector<3xi32>)>
  %2 = spv.GLSL.FrexpStruct %arg0 : vector<3xf32> -> !spv.struct<(vector<3xf32>, vector<3xi32>)>
  return
}

// -----

func.func @frexp_struct_mismatch_type(%arg0 : f32) -> () {
  // expected-error @+1 {{member zero of the resulting struct type must be the same type as the operand}}
  %2 = spv.GLSL.FrexpStruct %arg0 : f32 -> !spv.struct<(vector<3xf32>, i32)>
  return
}

// -----

func.func @frexp_struct_wrong_type(%arg0 : i32) -> () {
  // expected-error @+1 {{op operand #0 must be 16/32/64-bit float or vector of 16/32/64-bit float values}}
  %2 = spv.GLSL.FrexpStruct %arg0 : i32 -> !spv.struct<(i32, i32)>
  return
}

// -----

func.func @frexp_struct_mismatch_num_components(%arg0 : vector<3xf32>) -> () {
  // expected-error @+1 {{member one of the resulting struct type must have the same number of components as the operand type}}
  %2 = spv.GLSL.FrexpStruct %arg0 : vector<3xf32> -> !spv.struct<(vector<3xf32>, vector<2xi32>)>
  return
}

// -----

func.func @frexp_struct_not_i32(%arg0 : f32) -> () {
  // expected-error @+1 {{member one of the resulting struct type must be a scalar or vector of 32 bit integer type}}
  %2 = spv.GLSL.FrexpStruct %arg0 : f32 -> !spv.struct<(f32, i64)>
  return
}

// -----

//===----------------------------------------------------------------------===//
// spv.GLSL.Ldexp
//===----------------------------------------------------------------------===//

func.func @ldexp(%arg0 : f32, %arg1 : i32) -> () {
  // CHECK: {{%.*}} = spv.GLSL.Ldexp {{%.*}} : f32, {{%.*}} : i32 -> f32
  %0 = spv.GLSL.Ldexp %arg0 : f32, %arg1 : i32 -> f32
  return
}

// -----
func.func @ldexp_vec(%arg0 : vector<3xf32>, %arg1 : vector<3xi32>) -> () {
  // CHECK: {{%.*}} = spv.GLSL.Ldexp {{%.*}} : vector<3xf32>, {{%.*}} : vector<3xi32> -> vector<3xf32>
  %0 = spv.GLSL.Ldexp %arg0 : vector<3xf32>, %arg1 : vector<3xi32> -> vector<3xf32>
  return
}

// -----

func.func @ldexp_wrong_type_scalar(%arg0 : f32, %arg1 : vector<2xi32>) -> () {
  // expected-error @+1 {{operands must both be scalars or vectors}}
  %0 = spv.GLSL.Ldexp %arg0 : f32, %arg1 : vector<2xi32> -> f32
  return
}

// -----

func.func @ldexp_wrong_type_vec_1(%arg0 : vector<3xf32>, %arg1 : i32) -> () {
  // expected-error @+1 {{operands must both be scalars or vectors}}
  %0 = spv.GLSL.Ldexp %arg0 : vector<3xf32>, %arg1 : i32 -> vector<3xf32>
  return
}

// -----

func.func @ldexp_wrong_type_vec_2(%arg0 : vector<3xf32>, %arg1 : vector<2xi32>) -> () {
  // expected-error @+1 {{operands must have the same number of elements}}
  %0 = spv.GLSL.Ldexp %arg0 : vector<3xf32>, %arg1 : vector<2xi32> -> vector<3xf32>
  return
}

// -----

//===----------------------------------------------------------------------===//
// spv.GLSL.FMix
//===----------------------------------------------------------------------===//

func.func @fmix(%arg0 : f32, %arg1 : f32, %arg2 : f32) -> () {
  // CHECK: {{%.*}} = spv.GLSL.FMix {{%.*}} : f32, {{%.*}} : f32, {{%.*}} : f32 -> f32
  %0 = spv.GLSL.FMix %arg0 : f32, %arg1 : f32, %arg2 : f32 -> f32
  return
}

func.func @fmix_vector(%arg0 : vector<3xf32>, %arg1 : vector<3xf32>, %arg2 : vector<3xf32>) -> () {
  // CHECK: {{%.*}} = spv.GLSL.FMix {{%.*}} : vector<3xf32>, {{%.*}} : vector<3xf32>, {{%.*}} : vector<3xf32> -> vector<3xf32>
  %0 = spv.GLSL.FMix %arg0 : vector<3xf32>, %arg1 : vector<3xf32>, %arg2 : vector<3xf32> -> vector<3xf32>
  return
}

// -----

//===----------------------------------------------------------------------===//
// spv.GLSL.Exp
//===----------------------------------------------------------------------===//

func.func @findumsb(%arg0 : i32) -> () {
  // CHECK: spv.GLSL.FindUMsb {{%.*}} : i32
  %2 = spv.GLSL.FindUMsb %arg0 : i32
  return
}

func.func @findumsb_vector(%arg0 : vector<3xi32>) -> () {
  // CHECK: spv.GLSL.FindUMsb {{%.*}} : vector<3xi32>
  %2 = spv.GLSL.FindUMsb %arg0 : vector<3xi32>
  return
}

// -----

func.func @findumsb(%arg0 : i64) -> () {
  // expected-error @+1 {{operand #0 must be Int32 or vector of Int32}}
  %2 = spv.GLSL.FindUMsb %arg0 : i64
  return
}
