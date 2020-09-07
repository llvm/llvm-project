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
