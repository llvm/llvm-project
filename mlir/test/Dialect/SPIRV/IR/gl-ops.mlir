// RUN: mlir-opt -split-input-file -verify-diagnostics %s | FileCheck %s

//===----------------------------------------------------------------------===//
// spv.GL.Exp
//===----------------------------------------------------------------------===//

func.func @exp(%arg0 : f32) -> () {
  // CHECK: spv.GL.Exp {{%.*}} : f32
  %2 = spv.GL.Exp %arg0 : f32
  return
}

func.func @expvec(%arg0 : vector<3xf16>) -> () {
  // CHECK: spv.GL.Exp {{%.*}} : vector<3xf16>
  %2 = spv.GL.Exp %arg0 : vector<3xf16>
  return
}

// -----

func.func @exp(%arg0 : i32) -> () {
  // expected-error @+1 {{op operand #0 must be 16/32-bit float or vector of 16/32-bit float values}}
  %2 = spv.GL.Exp %arg0 : i32
  return
}

// -----

func.func @exp(%arg0 : vector<5xf32>) -> () {
  // expected-error @+1 {{op operand #0 must be 16/32-bit float or vector of 16/32-bit float values of length 2/3/4}}
  %2 = spv.GL.Exp %arg0 : vector<5xf32>
  return
}

// -----

func.func @exp(%arg0 : f32, %arg1 : f32) -> () {
  // expected-error @+1 {{expected ':'}}
  %2 = spv.GL.Exp %arg0, %arg1 : i32
  return
}

// -----

func.func @exp(%arg0 : i32) -> () {
  // expected-error @+1 {{expected non-function type}}
  %2 = spv.GL.Exp %arg0 :
  return
}

// -----

//===----------------------------------------------------------------------===//
// spv.GL.{F|S|U}{Max|Min}
//===----------------------------------------------------------------------===//

func.func @fmaxmin(%arg0 : f32, %arg1 : f32) {
  // CHECK: spv.GL.FMax {{%.*}}, {{%.*}} : f32
  %1 = spv.GL.FMax %arg0, %arg1 : f32
  // CHECK: spv.GL.FMin {{%.*}}, {{%.*}} : f32
  %2 = spv.GL.FMin %arg0, %arg1 : f32
  return
}

func.func @fmaxminvec(%arg0 : vector<3xf16>, %arg1 : vector<3xf16>) {
  // CHECK: spv.GL.FMax {{%.*}}, {{%.*}} : vector<3xf16>
  %1 = spv.GL.FMax %arg0, %arg1 : vector<3xf16>
  // CHECK: spv.GL.FMin {{%.*}}, {{%.*}} : vector<3xf16>
  %2 = spv.GL.FMin %arg0, %arg1 : vector<3xf16>
  return
}

func.func @fmaxminf64(%arg0 : f64, %arg1 : f64) {
  // CHECK: spv.GL.FMax {{%.*}}, {{%.*}} : f64
  %1 = spv.GL.FMax %arg0, %arg1 : f64
  // CHECK: spv.GL.FMin {{%.*}}, {{%.*}} : f64
  %2 = spv.GL.FMin %arg0, %arg1 : f64
  return
}

func.func @iminmax(%arg0: i32, %arg1: i32) {
  // CHECK: spv.GL.SMax {{%.*}}, {{%.*}} : i32
  %1 = spv.GL.SMax %arg0, %arg1 : i32
  // CHECK: spv.GL.UMax {{%.*}}, {{%.*}} : i32
  %2 = spv.GL.UMax %arg0, %arg1 : i32
  // CHECK: spv.GL.SMin {{%.*}}, {{%.*}} : i32
  %3 = spv.GL.SMin %arg0, %arg1 : i32
  // CHECK: spv.GL.UMin {{%.*}}, {{%.*}} : i32
  %4 = spv.GL.UMin %arg0, %arg1 : i32
  return
}

// -----

//===----------------------------------------------------------------------===//
// spv.GL.InverseSqrt
//===----------------------------------------------------------------------===//

func.func @inversesqrt(%arg0 : f32) -> () {
  // CHECK: spv.GL.InverseSqrt {{%.*}} : f32
  %2 = spv.GL.InverseSqrt %arg0 : f32
  return
}

func.func @inversesqrtvec(%arg0 : vector<3xf16>) -> () {
  // CHECK: spv.GL.InverseSqrt {{%.*}} : vector<3xf16>
  %2 = spv.GL.InverseSqrt %arg0 : vector<3xf16>
  return
}

// -----

//===----------------------------------------------------------------------===//
// spv.GL.Sqrt
//===----------------------------------------------------------------------===//

func.func @sqrt(%arg0 : f32) -> () {
  // CHECK: spv.GL.Sqrt {{%.*}} : f32
  %2 = spv.GL.Sqrt %arg0 : f32
  return
}

func.func @sqrtvec(%arg0 : vector<3xf16>) -> () {
  // CHECK: spv.GL.Sqrt {{%.*}} : vector<3xf16>
  %2 = spv.GL.Sqrt %arg0 : vector<3xf16>
  return
}

//===----------------------------------------------------------------------===//
// spv.GL.Cos
//===----------------------------------------------------------------------===//

func.func @cos(%arg0 : f32) -> () {
  // CHECK: spv.GL.Cos {{%.*}} : f32
  %2 = spv.GL.Cos %arg0 : f32
  return
}

func.func @cosvec(%arg0 : vector<3xf16>) -> () {
  // CHECK: spv.GL.Cos {{%.*}} : vector<3xf16>
  %2 = spv.GL.Cos %arg0 : vector<3xf16>
  return
}

//===----------------------------------------------------------------------===//
// spv.GL.Sin
//===----------------------------------------------------------------------===//

func.func @sin(%arg0 : f32) -> () {
  // CHECK: spv.GL.Sin {{%.*}} : f32
  %2 = spv.GL.Sin %arg0 : f32
  return
}

func.func @sinvec(%arg0 : vector<3xf16>) -> () {
  // CHECK: spv.GL.Sin {{%.*}} : vector<3xf16>
  %2 = spv.GL.Sin %arg0 : vector<3xf16>
  return
}

//===----------------------------------------------------------------------===//
// spv.GL.Tan
//===----------------------------------------------------------------------===//

func.func @tan(%arg0 : f32) -> () {
  // CHECK: spv.GL.Tan {{%.*}} : f32
  %2 = spv.GL.Tan %arg0 : f32
  return
}

func.func @tanvec(%arg0 : vector<3xf16>) -> () {
  // CHECK: spv.GL.Tan {{%.*}} : vector<3xf16>
  %2 = spv.GL.Tan %arg0 : vector<3xf16>
  return
}

//===----------------------------------------------------------------------===//
// spv.GL.Acos
//===----------------------------------------------------------------------===//

func.func @acos(%arg0 : f32) -> () {
  // CHECK: spv.GL.Acos {{%.*}} : f32
  %2 = spv.GL.Acos %arg0 : f32
  return
}

func.func @acosvec(%arg0 : vector<3xf16>) -> () {
  // CHECK: spv.GL.Acos {{%.*}} : vector<3xf16>
  %2 = spv.GL.Acos %arg0 : vector<3xf16>
  return
}

//===----------------------------------------------------------------------===//
// spv.GL.Asin
//===----------------------------------------------------------------------===//

func.func @asin(%arg0 : f32) -> () {
  // CHECK: spv.GL.Asin {{%.*}} : f32
  %2 = spv.GL.Asin %arg0 : f32
  return
}

func.func @asinvec(%arg0 : vector<3xf16>) -> () {
  // CHECK: spv.GL.Asin {{%.*}} : vector<3xf16>
  %2 = spv.GL.Asin %arg0 : vector<3xf16>
  return
}

//===----------------------------------------------------------------------===//
// spv.GL.Atan
//===----------------------------------------------------------------------===//

func.func @atan(%arg0 : f32) -> () {
  // CHECK: spv.GL.Atan {{%.*}} : f32
  %2 = spv.GL.Atan %arg0 : f32
  return
}

func.func @atanvec(%arg0 : vector<3xf16>) -> () {
  // CHECK: spv.GL.Atan {{%.*}} : vector<3xf16>
  %2 = spv.GL.Atan %arg0 : vector<3xf16>
  return
}

//===----------------------------------------------------------------------===//
// spv.GL.Sinh
//===----------------------------------------------------------------------===//

func.func @sinh(%arg0 : f32) -> () {
  // CHECK: spv.GL.Sinh {{%.*}} : f32
  %2 = spv.GL.Sinh %arg0 : f32
  return
}

func.func @sinhvec(%arg0 : vector<3xf16>) -> () {
  // CHECK: spv.GL.Sinh {{%.*}} : vector<3xf16>
  %2 = spv.GL.Sinh %arg0 : vector<3xf16>
  return
}

//===----------------------------------------------------------------------===//
// spv.GL.Cosh
//===----------------------------------------------------------------------===//

func.func @cosh(%arg0 : f32) -> () {
  // CHECK: spv.GL.Cosh {{%.*}} : f32
  %2 = spv.GL.Cosh %arg0 : f32
  return
}

func.func @coshvec(%arg0 : vector<3xf16>) -> () {
  // CHECK: spv.GL.Cosh {{%.*}} : vector<3xf16>
  %2 = spv.GL.Cosh %arg0 : vector<3xf16>
  return
}

//===----------------------------------------------------------------------===//
// spv.GL.Pow
//===----------------------------------------------------------------------===//

func.func @pow(%arg0 : f32, %arg1 : f32) -> () {
  // CHECK: spv.GL.Pow {{%.*}}, {{%.*}} : f32
  %2 = spv.GL.Pow %arg0, %arg1 : f32
  return
}

func.func @powvec(%arg0 : vector<3xf16>, %arg1 : vector<3xf16>) -> () {
  // CHECK: spv.GL.Pow {{%.*}}, {{%.*}} : vector<3xf16>
  %2 = spv.GL.Pow %arg0, %arg1 : vector<3xf16>
  return
}

// -----

//===----------------------------------------------------------------------===//
// spv.GL.Round
//===----------------------------------------------------------------------===//

func.func @round(%arg0 : f32) -> () {
  // CHECK: spv.GL.Round {{%.*}} : f32
  %2 = spv.GL.Round %arg0 : f32
  return
}

func.func @roundvec(%arg0 : vector<3xf16>) -> () {
  // CHECK: spv.GL.Round {{%.*}} : vector<3xf16>
  %2 = spv.GL.Round %arg0 : vector<3xf16>
  return
}

// -----

//===----------------------------------------------------------------------===//
// spv.GL.FClamp
//===----------------------------------------------------------------------===//

func.func @fclamp(%arg0 : f32, %min : f32, %max : f32) -> () {
  // CHECK: spv.GL.FClamp {{%[^,]*}}, {{%[^,]*}}, {{%[^,]*}} : f32
  %2 = spv.GL.FClamp %arg0, %min, %max : f32
  return
}

// -----

func.func @fclamp(%arg0 : vector<3xf32>, %min : vector<3xf32>, %max : vector<3xf32>) -> () {
  // CHECK: spv.GL.FClamp {{%[^,]*}}, {{%[^,]*}}, {{%[^,]*}} : vector<3xf32>
  %2 = spv.GL.FClamp %arg0, %min, %max : vector<3xf32>
  return
}

// -----

//===----------------------------------------------------------------------===//
// spv.GL.UClamp
//===----------------------------------------------------------------------===//

func.func @uclamp(%arg0 : ui32, %min : ui32, %max : ui32) -> () {
  // CHECK: spv.GL.UClamp {{%[^,]*}}, {{%[^,]*}}, {{%[^,]*}} : ui32
  %2 = spv.GL.UClamp %arg0, %min, %max : ui32
  return
}

// -----

func.func @uclamp(%arg0 : vector<4xi32>, %min : vector<4xi32>, %max : vector<4xi32>) -> () {
  // CHECK: spv.GL.UClamp {{%[^,]*}}, {{%[^,]*}}, {{%[^,]*}} : vector<4xi32>
  %2 = spv.GL.UClamp %arg0, %min, %max : vector<4xi32>
  return
}

// -----

func.func @uclamp(%arg0 : si32, %min : si32, %max : si32) -> () {
  // CHECK: spv.GL.UClamp
  %2 = spv.GL.UClamp %arg0, %min, %max : si32
  return
}

// -----

//===----------------------------------------------------------------------===//
// spv.GL.SClamp
//===----------------------------------------------------------------------===//

func.func @sclamp(%arg0 : si32, %min : si32, %max : si32) -> () {
  // CHECK: spv.GL.SClamp {{%[^,]*}}, {{%[^,]*}}, {{%[^,]*}} : si32
  %2 = spv.GL.SClamp %arg0, %min, %max : si32
  return
}

// -----

func.func @sclamp(%arg0 : vector<4xsi32>, %min : vector<4xsi32>, %max : vector<4xsi32>) -> () {
  // CHECK: spv.GL.SClamp {{%[^,]*}}, {{%[^,]*}}, {{%[^,]*}} : vector<4xsi32>
  %2 = spv.GL.SClamp %arg0, %min, %max : vector<4xsi32>
  return
}

// -----

func.func @sclamp(%arg0 : i32, %min : i32, %max : i32) -> () {
  // CHECK: spv.GL.SClamp
  %2 = spv.GL.SClamp %arg0, %min, %max : i32
  return
}

// -----

//===----------------------------------------------------------------------===//
// spv.GL.Fma
//===----------------------------------------------------------------------===//

func.func @fma(%a : f32, %b : f32, %c : f32) -> () {
  // CHECK: spv.GL.Fma {{%[^,]*}}, {{%[^,]*}}, {{%[^,]*}} : f32
  %2 = spv.GL.Fma %a, %b, %c : f32
  return
}

// -----

func.func @fma(%a : vector<3xf32>, %b : vector<3xf32>, %c : vector<3xf32>) -> () {
  // CHECK: spv.GL.Fma {{%[^,]*}}, {{%[^,]*}}, {{%[^,]*}} : vector<3xf32>
  %2 = spv.GL.Fma %a, %b, %c : vector<3xf32>
  return
}
// -----

//===----------------------------------------------------------------------===//
// spv.GL.FrexpStruct
//===----------------------------------------------------------------------===//

func.func @frexp_struct(%arg0 : f32) -> () {
  // CHECK: spv.GL.FrexpStruct {{%.*}} : f32 -> !spv.struct<(f32, i32)>
  %2 = spv.GL.FrexpStruct %arg0 : f32 -> !spv.struct<(f32, i32)>
  return
}

func.func @frexp_struct_64(%arg0 : f64) -> () {
  // CHECK: spv.GL.FrexpStruct {{%.*}} : f64 -> !spv.struct<(f64, i32)>
  %2 = spv.GL.FrexpStruct %arg0 : f64 -> !spv.struct<(f64, i32)>
  return
}

func.func @frexp_struct_vec(%arg0 : vector<3xf32>) -> () {
  // CHECK: spv.GL.FrexpStruct {{%.*}} : vector<3xf32> -> !spv.struct<(vector<3xf32>, vector<3xi32>)>
  %2 = spv.GL.FrexpStruct %arg0 : vector<3xf32> -> !spv.struct<(vector<3xf32>, vector<3xi32>)>
  return
}

// -----

func.func @frexp_struct_mismatch_type(%arg0 : f32) -> () {
  // expected-error @+1 {{member zero of the resulting struct type must be the same type as the operand}}
  %2 = spv.GL.FrexpStruct %arg0 : f32 -> !spv.struct<(vector<3xf32>, i32)>
  return
}

// -----

func.func @frexp_struct_wrong_type(%arg0 : i32) -> () {
  // expected-error @+1 {{op operand #0 must be 16/32/64-bit float or vector of 16/32/64-bit float values}}
  %2 = spv.GL.FrexpStruct %arg0 : i32 -> !spv.struct<(i32, i32)>
  return
}

// -----

func.func @frexp_struct_mismatch_num_components(%arg0 : vector<3xf32>) -> () {
  // expected-error @+1 {{member one of the resulting struct type must have the same number of components as the operand type}}
  %2 = spv.GL.FrexpStruct %arg0 : vector<3xf32> -> !spv.struct<(vector<3xf32>, vector<2xi32>)>
  return
}

// -----

func.func @frexp_struct_not_i32(%arg0 : f32) -> () {
  // expected-error @+1 {{member one of the resulting struct type must be a scalar or vector of 32 bit integer type}}
  %2 = spv.GL.FrexpStruct %arg0 : f32 -> !spv.struct<(f32, i64)>
  return
}

// -----

//===----------------------------------------------------------------------===//
// spv.GL.Ldexp
//===----------------------------------------------------------------------===//

func.func @ldexp(%arg0 : f32, %arg1 : i32) -> () {
  // CHECK: {{%.*}} = spv.GL.Ldexp {{%.*}} : f32, {{%.*}} : i32 -> f32
  %0 = spv.GL.Ldexp %arg0 : f32, %arg1 : i32 -> f32
  return
}

// -----
func.func @ldexp_vec(%arg0 : vector<3xf32>, %arg1 : vector<3xi32>) -> () {
  // CHECK: {{%.*}} = spv.GL.Ldexp {{%.*}} : vector<3xf32>, {{%.*}} : vector<3xi32> -> vector<3xf32>
  %0 = spv.GL.Ldexp %arg0 : vector<3xf32>, %arg1 : vector<3xi32> -> vector<3xf32>
  return
}

// -----

func.func @ldexp_wrong_type_scalar(%arg0 : f32, %arg1 : vector<2xi32>) -> () {
  // expected-error @+1 {{operands must both be scalars or vectors}}
  %0 = spv.GL.Ldexp %arg0 : f32, %arg1 : vector<2xi32> -> f32
  return
}

// -----

func.func @ldexp_wrong_type_vec_1(%arg0 : vector<3xf32>, %arg1 : i32) -> () {
  // expected-error @+1 {{operands must both be scalars or vectors}}
  %0 = spv.GL.Ldexp %arg0 : vector<3xf32>, %arg1 : i32 -> vector<3xf32>
  return
}

// -----

func.func @ldexp_wrong_type_vec_2(%arg0 : vector<3xf32>, %arg1 : vector<2xi32>) -> () {
  // expected-error @+1 {{operands must have the same number of elements}}
  %0 = spv.GL.Ldexp %arg0 : vector<3xf32>, %arg1 : vector<2xi32> -> vector<3xf32>
  return
}

// -----

//===----------------------------------------------------------------------===//
// spv.GL.FMix
//===----------------------------------------------------------------------===//

func.func @fmix(%arg0 : f32, %arg1 : f32, %arg2 : f32) -> () {
  // CHECK: {{%.*}} = spv.GL.FMix {{%.*}} : f32, {{%.*}} : f32, {{%.*}} : f32 -> f32
  %0 = spv.GL.FMix %arg0 : f32, %arg1 : f32, %arg2 : f32 -> f32
  return
}

func.func @fmix_vector(%arg0 : vector<3xf32>, %arg1 : vector<3xf32>, %arg2 : vector<3xf32>) -> () {
  // CHECK: {{%.*}} = spv.GL.FMix {{%.*}} : vector<3xf32>, {{%.*}} : vector<3xf32>, {{%.*}} : vector<3xf32> -> vector<3xf32>
  %0 = spv.GL.FMix %arg0 : vector<3xf32>, %arg1 : vector<3xf32>, %arg2 : vector<3xf32> -> vector<3xf32>
  return
}

// -----

//===----------------------------------------------------------------------===//
// spv.GL.Exp
//===----------------------------------------------------------------------===//

func.func @findumsb(%arg0 : i32) -> () {
  // CHECK: spv.GL.FindUMsb {{%.*}} : i32
  %2 = spv.GL.FindUMsb %arg0 : i32
  return
}

func.func @findumsb_vector(%arg0 : vector<3xi32>) -> () {
  // CHECK: spv.GL.FindUMsb {{%.*}} : vector<3xi32>
  %2 = spv.GL.FindUMsb %arg0 : vector<3xi32>
  return
}

// -----

func.func @findumsb(%arg0 : i64) -> () {
  // expected-error @+1 {{operand #0 must be Int32 or vector of Int32}}
  %2 = spv.GL.FindUMsb %arg0 : i64
  return
}
