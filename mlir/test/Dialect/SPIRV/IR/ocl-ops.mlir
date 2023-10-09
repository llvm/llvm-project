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

func.func @exp_any_vec(%arg0 : vector<5xf32>) -> () {
  // CHECK: spirv.CL.exp {{%.*}} : vector<5xf32>
  %2 = spirv.CL.exp %arg0 : vector<5xf32>
  return
}

// -----

func.func @exp(%arg0 : i32) -> () {
  // expected-error @+1 {{op operand #0 must be 16/32/64-bit float or vector of 16/32/64-bit float values}}
  %2 = spirv.CL.exp %arg0 : i32
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

// -----

func.func @fabs_any_vec(%arg0 : vector<5xf32>) -> () {
  // CHECK: spirv.CL.fabs {{%.*}} : vector<5xf32>
  %2 = spirv.CL.fabs %arg0 : vector<5xf32>
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

// -----

func.func @sabs_any_vec(%arg0 : vector<5xi32>) -> () {
  // CHECK: spirv.CL.s_abs {{%.*}} : vector<5xi32>
  %2 = spirv.CL.s_abs %arg0 : vector<5xi32>
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
func.func @printf(%arg0 : !spirv.ptr<i8, UniformConstant>, %arg1 : i32, %arg2 : i32) -> i32 {
  // CHECK: spirv.CL.printf {{%.*}}, {{%.*}}, {{%.*}} : (!spirv.ptr<i8, UniformConstant>, (i32, i32)) -> i32
  %0 = spirv.CL.printf %arg0, %arg1, %arg2 : (!spirv.ptr<i8, UniformConstant>, (i32, i32)) -> i32
  return %0 : i32
}

