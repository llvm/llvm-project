// RUN: mlir-opt -split-input-file -verify-diagnostics %s | FileCheck %s

//===----------------------------------------------------------------------===//
// spirv.CL.exp
//===----------------------------------------------------------------------===//

func.func @exp(%arg0 : f32) -> () {
  // CHECK: spirv.CL.exp {{%.*}} : f32
  %2 = spirv.CL.exp %arg0 : f32
  return
}

func.func @expvec(%arg0 : vector<3xf16>) -> () {
  // CHECK: spirv.CL.exp {{%.*}} : vector<3xf16>
  %2 = spirv.CL.exp %arg0 : vector<3xf16>
  return
}

// -----

func.func @exp(%arg0 : i32) -> () {
  // expected-error @+1 {{op operand #0 must be 16/32/64-bit float or vector of 16/32/64-bit float values}}
  %2 = spirv.CL.exp %arg0 : i32
  return
}

// -----

func.func @exp(%arg0 : vector<5xf32>) -> () {
  // expected-error @+1 {{op operand #0 must be 16/32/64-bit float or vector of 16/32/64-bit float values of length 2/3/4}}
  %2 = spirv.CL.exp %arg0 : vector<5xf32>
  return
}

// -----

func.func @exp(%arg0 : f32, %arg1 : f32) -> () {
  // expected-error @+1 {{expected ':'}}
  %2 = spirv.CL.exp %arg0, %arg1 : i32
  return
}

// -----

func.func @exp(%arg0 : i32) -> () {
  // expected-error @+1 {{expected non-function type}}
  %2 = spirv.CL.exp %arg0 :
  return
}

// -----

//===----------------------------------------------------------------------===//
// spirv.CL.fabs
//===----------------------------------------------------------------------===//

func.func @fabs(%arg0 : f32) -> () {
  // CHECK: spirv.CL.fabs {{%.*}} : f32
  %2 = spirv.CL.fabs %arg0 : f32
  return
}

func.func @fabsvec(%arg0 : vector<3xf16>) -> () {
  // CHECK: spirv.CL.fabs {{%.*}} : vector<3xf16>
  %2 = spirv.CL.fabs %arg0 : vector<3xf16>
  return
}

func.func @fabsf64(%arg0 : f64) -> () {
  // CHECK: spirv.CL.fabs {{%.*}} : f64
  %2 = spirv.CL.fabs %arg0 : f64
  return
}

// -----

func.func @fabs(%arg0 : i32) -> () {
  // expected-error @+1 {{op operand #0 must be 16/32/64-bit float or vector of 16/32/64-bit float values}}
  %2 = spirv.CL.fabs %arg0 : i32
  return
}

// -----

func.func @fabs(%arg0 : vector<5xf32>) -> () {
  // expected-error @+1 {{op operand #0 must be 16/32/64-bit float or vector of 16/32/64-bit float values of length 2/3/4}}
  %2 = spirv.CL.fabs %arg0 : vector<5xf32>
  return
}

// -----

func.func @fabs(%arg0 : f32, %arg1 : f32) -> () {
  // expected-error @+1 {{expected ':'}}
  %2 = spirv.CL.fabs %arg0, %arg1 : i32
  return
}

// -----

func.func @fabs(%arg0 : i32) -> () {
  // expected-error @+1 {{expected non-function type}}
  %2 = spirv.CL.fabs %arg0 :
  return
}

// -----

//===----------------------------------------------------------------------===//
// spirv.CL.s_abs
//===----------------------------------------------------------------------===//

func.func @sabs(%arg0 : i32) -> () {
  // CHECK: spirv.CL.s_abs {{%.*}} : i32
  %2 = spirv.CL.s_abs %arg0 : i32
  return
}

func.func @sabsvec(%arg0 : vector<3xi16>) -> () {
  // CHECK: spirv.CL.s_abs {{%.*}} : vector<3xi16>
  %2 = spirv.CL.s_abs %arg0 : vector<3xi16>
  return
}

func.func @sabsi64(%arg0 : i64) -> () {
  // CHECK: spirv.CL.s_abs {{%.*}} : i64
  %2 = spirv.CL.s_abs %arg0 : i64
  return
}

func.func @sabsi8(%arg0 : i8) -> () {
  // CHECK: spirv.CL.s_abs {{%.*}} : i8
  %2 = spirv.CL.s_abs %arg0 : i8
  return
}

// -----

func.func @sabs(%arg0 : f32) -> () {
  // expected-error @+1 {{op operand #0 must be 8/16/32/64-bit integer or vector of 8/16/32/64-bit integer values}}
  %2 = spirv.CL.s_abs %arg0 : f32
  return
}

// -----

func.func @sabs(%arg0 : vector<5xi32>) -> () {
  // expected-error @+1 {{op operand #0 must be 8/16/32/64-bit integer or vector of 8/16/32/64-bit integer values of length 2/3/4}}
  %2 = spirv.CL.s_abs %arg0 : vector<5xi32>
  return
}

// -----

func.func @sabs(%arg0 : i32, %arg1 : i32) -> () {
  // expected-error @+1 {{expected ':'}}
  %2 = spirv.CL.s_abs %arg0, %arg1 : i32
  return
}

// -----

func.func @sabs(%arg0 : i32) -> () {
  // expected-error @+1 {{expected non-function type}}
  %2 = spirv.CL.s_abs %arg0 :
  return
}

// -----

//===----------------------------------------------------------------------===//
// spirv.CL.fma
//===----------------------------------------------------------------------===//

func.func @fma(%a : f32, %b : f32, %c : f32) -> () {
  // CHECK: spirv.CL.fma {{%[^,]*}}, {{%[^,]*}}, {{%[^,]*}} : f32
  %2 = spirv.CL.fma %a, %b, %c : f32
  return
}

// -----

func.func @fma(%a : vector<3xf32>, %b : vector<3xf32>, %c : vector<3xf32>) -> () {
  // CHECK: spirv.CL.fma {{%[^,]*}}, {{%[^,]*}}, {{%[^,]*}} : vector<3xf32>
  %2 = spirv.CL.fma %a, %b, %c : vector<3xf32>
  return
}

// -----

//===----------------------------------------------------------------------===//
// spirv.CL.mix
//===----------------------------------------------------------------------===//

func.func @mix(%a : f32, %b : f32, %c : f32) -> () {
  // CHECK: spirv.CL.mix {{%[^,]*}}, {{%[^,]*}}, {{%[^,]*}} : f32
  %2 = spirv.CL.mix %a, %b, %c : f32
  return
}

// -----

func.func @mix(%a : vector<3xf32>, %b : vector<3xf32>, %c : vector<3xf32>) -> () {
  // CHECK: spirv.CL.mix {{%[^,]*}}, {{%[^,]*}}, {{%[^,]*}} : vector<3xf32>
  %2 = spirv.CL.mix %a, %b, %c : vector<3xf32>
  return
}

// -----

//===----------------------------------------------------------------------===//
// spirv.CL.{F|S|U}{Max|Min}
//===----------------------------------------------------------------------===//

func.func @fmaxmin(%arg0 : f32, %arg1 : f32) {
  // CHECK: spirv.CL.fmax {{%.*}}, {{%.*}} : f32
  %1 = spirv.CL.fmax %arg0, %arg1 : f32
  // CHECK: spirv.CL.fmin {{%.*}}, {{%.*}} : f32
  %2 = spirv.CL.fmin %arg0, %arg1 : f32
  return
}

func.func @fmaxminvec(%arg0 : vector<3xf16>, %arg1 : vector<3xf16>) {
  // CHECK: spirv.CL.fmax {{%.*}}, {{%.*}} : vector<3xf16>
  %1 = spirv.CL.fmax %arg0, %arg1 : vector<3xf16>
  // CHECK: spirv.CL.fmin {{%.*}}, {{%.*}} : vector<3xf16>
  %2 = spirv.CL.fmin %arg0, %arg1 : vector<3xf16>
  return
}

func.func @fmaxminf64(%arg0 : f64, %arg1 : f64) {
  // CHECK: spirv.CL.fmax {{%.*}}, {{%.*}} : f64
  %1 = spirv.CL.fmax %arg0, %arg1 : f64
  // CHECK: spirv.CL.fmin {{%.*}}, {{%.*}} : f64
  %2 = spirv.CL.fmin %arg0, %arg1 : f64
  return
}

func.func @iminmax(%arg0: i32, %arg1: i32) {
  // CHECK: spirv.CL.s_max {{%.*}}, {{%.*}} : i32
  %1 = spirv.CL.s_max %arg0, %arg1 : i32
  // CHECK: spirv.CL.u_max {{%.*}}, {{%.*}} : i32
  %2 = spirv.CL.u_max %arg0, %arg1 : i32
  // CHECK: spirv.CL.s_min {{%.*}}, {{%.*}} : i32
  %3 = spirv.CL.s_min %arg0, %arg1 : i32
  // CHECK: spirv.CL.u_min {{%.*}}, {{%.*}} : i32
  %4 = spirv.CL.u_min %arg0, %arg1 : i32
  return
}

// -----

//===----------------------------------------------------------------------===//
// spirv.CL.rint
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @rint(
func.func @rint(%arg0 : f32) -> () {
  // CHECK: spirv.CL.rint {{%.*}} : f32
  %0 = spirv.CL.rint %arg0 : f32
  return
}

// CHECK-LABEL: func.func @rintvec(
func.func @rintvec(%arg0 : vector<3xf16>) -> () {
  // CHECK: spirv.CL.rint {{%.*}} : vector<3xf16>
  %0 = spirv.CL.rint %arg0 : vector<3xf16>
  return
}

// -----

//===----------------------------------------------------------------------===//
// spirv.CL.printf
//===----------------------------------------------------------------------===//
// CHECK-LABEL: func.func @printf(
func.func @printf(%fmt : !spirv.ptr<i8, UniformConstant>, %arg1 : i32, %arg2 : i32) -> i32 {
  // CHECK: spirv.CL.printf {{%.*}} {{%.*}}, {{%.*}} : !spirv.ptr<i8, UniformConstant>, i32, i32 -> i32
  %0 = spirv.CL.printf %fmt %arg1, %arg2 : !spirv.ptr<i8, UniformConstant>, i32, i32 -> i32
  return %0 : i32
}

// -----

func.func @tan(%arg0 : f32) -> () {
  // CHECK: spirv.CL.tan {{%.*}} : f32
  %2 = spirv.CL.tan %arg0 : f32
  return
}

// -----

func.func @tan(%arg0 : vector<4xf16>) -> () {
  // CHECK: spirv.CL.tan {{%.*}} : vector<4xf16>
  %2 = spirv.CL.tan %arg0 : vector<4xf16>
  return
}

// -----

func.func @atan(%arg0 : f32) -> () {
  // CHECK: spirv.CL.atan {{%.*}} : f32
  %2 = spirv.CL.atan %arg0 : f32
  return
}

// -----

func.func @atan(%arg0 : vector<4xf16>) -> () {
  // CHECK: spirv.CL.atan {{%.*}} : vector<4xf16>
  %2 = spirv.CL.atan %arg0 : vector<4xf16>
  return
}

// -----

func.func @atanh(%arg0 : f32) -> () {
  // CHECK: spirv.CL.atanh {{%.*}} : f32
  %2 = spirv.CL.atanh %arg0 : f32
  return
}

// -----

func.func @atanh(%arg0 : vector<4xf16>) -> () {
  // CHECK: spirv.CL.atanh {{%.*}} : vector<4xf16>
  %2 = spirv.CL.atanh %arg0 : vector<4xf16>
  return
}

// -----

func.func @sinh(%arg0 : f32) -> () {
  // CHECK: spirv.CL.sinh {{%.*}} : f32
  %2 = spirv.CL.sinh %arg0 : f32
  return
}

// -----

func.func @sinh(%arg0 : vector<4xf16>) -> () {
  // CHECK: spirv.CL.sinh {{%.*}} : vector<4xf16>
  %2 = spirv.CL.sinh %arg0 : vector<4xf16>
  return
}

// -----

func.func @cosh(%arg0 : f32) -> () {
  // CHECK: spirv.CL.cosh {{%.*}} : f32
  %2 = spirv.CL.cosh %arg0 : f32
  return
}

// -----

func.func @cosh(%arg0 : vector<4xf16>) -> () {
  // CHECK: spirv.CL.cosh {{%.*}} : vector<4xf16>
  %2 = spirv.CL.cosh %arg0 : vector<4xf16>
  return
}

// -----

func.func @asin(%arg0 : f32) -> () {
  // CHECK: spirv.CL.asin {{%.*}} : f32
  %2 = spirv.CL.asin %arg0 : f32
  return
}

// -----

func.func @asin(%arg0 : vector<4xf16>) -> () {
  // CHECK: spirv.CL.asin {{%.*}} : vector<4xf16>
  %2 = spirv.CL.asin %arg0 : vector<4xf16>
  return
}

// -----

func.func @asinh(%arg0 : f32) -> () {
  // CHECK: spirv.CL.asinh {{%.*}} : f32
  %2 = spirv.CL.asinh %arg0 : f32
  return
}

// -----

func.func @asinh(%arg0 : vector<4xf16>) -> () {
  // CHECK: spirv.CL.asinh {{%.*}} : vector<4xf16>
  %2 = spirv.CL.asinh %arg0 : vector<4xf16>
  return
}

// -----

func.func @acos(%arg0 : f32) -> () {
  // CHECK: spirv.CL.acos {{%.*}} : f32
  %2 = spirv.CL.acos %arg0 : f32
  return
}

// -----

func.func @acos(%arg0 : vector<4xf16>) -> () {
  // CHECK: spirv.CL.acos {{%.*}} : vector<4xf16>
  %2 = spirv.CL.acos %arg0 : vector<4xf16>
  return
}

// -----

func.func @acosh(%arg0 : f32) -> () {
  // CHECK: spirv.CL.acosh {{%.*}} : f32
  %2 = spirv.CL.acosh %arg0 : f32
  return
}

// -----

func.func @acosh(%arg0 : vector<4xf16>) -> () {
  // CHECK: spirv.CL.acosh {{%.*}} : vector<4xf16>
  %2 = spirv.CL.acosh %arg0 : vector<4xf16>
  return
}

// -----

func.func @atan2(%arg0 : f32, %arg1 : f32) -> () {
  // CHECK: spirv.CL.atan2 {{%.*}}, {{%.*}} : f32
  %2 = spirv.CL.atan2 %arg0, %arg1 : f32
  return
}

// -----

func.func @atan2(%arg0 : vector<4xf16>, %arg1 : vector<4xf16>) -> () {
  // CHECK: spirv.CL.atan2 {{%.*}}, {{%.*}} : vector<4xf16>
  %2 = spirv.CL.atan2 %arg0, %arg1 : vector<4xf16>
  return
}

