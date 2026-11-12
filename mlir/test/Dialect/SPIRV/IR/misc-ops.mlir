// RUN: mlir-opt -split-input-file -verify-diagnostics %s | FileCheck %s

//===----------------------------------------------------------------------===//
// spirv.Undef
//===----------------------------------------------------------------------===//

func.func @undef() -> () {
  // CHECK: %{{.*}} = spirv.Undef : f32
  %0 = spirv.Undef : f32
  // CHECK: %{{.*}} = spirv.Undef : vector<4xf32>
  %1 = spirv.Undef : vector<4xf32>
  spirv.Return
}

// -----

func.func @undef() -> () {
  // expected-error @+1{{expected non-function type}}
  %0 = spirv.Undef :
  spirv.Return
}

// -----

func.func @undef() -> () {
  // expected-error @+1{{expected ':'}}
  %0 = spirv.Undef
  spirv.Return
}

// -----

func.func @assume_true(%arg : i1) -> () {
  // CHECK: spirv.KHR.AssumeTrue %{{.*}}
  spirv.KHR.AssumeTrue %arg
  spirv.Return
}

// -----

func.func @assume_true(%arg : f32) -> () {
  // expected-error @+2{{use of value '%arg' expects different type than prior uses: 'i1' vs 'f32'}}
  // expected-note @-2 {{prior use here}}
  spirv.KHR.AssumeTrue %arg
  spirv.Return
}

// -----

//===----------------------------------------------------------------------===//
// spirv.KHR.Expect
//===----------------------------------------------------------------------===//

func.func @expect_scalar_int(%val : i32, %expected : i32) -> i32 {
  // CHECK: %{{.*}} = spirv.KHR.Expect %{{.*}}, %{{.*}} : i32
  %0 = spirv.KHR.Expect %val, %expected : i32
  spirv.ReturnValue %0 : i32
}

// -----

func.func @expect_scalar_bool(%val : i1, %expected : i1) -> i1 {
  // CHECK: %{{.*}} = spirv.KHR.Expect %{{.*}}, %{{.*}} : i1
  %0 = spirv.KHR.Expect %val, %expected : i1
  spirv.ReturnValue %0 : i1
}

// -----

func.func @expect_vector_int(%val : vector<4xi32>, %expected : vector<4xi32>) -> vector<4xi32> {
  // CHECK: %{{.*}} = spirv.KHR.Expect %{{.*}}, %{{.*}} : vector<4xi32>
  %0 = spirv.KHR.Expect %val, %expected : vector<4xi32>
  spirv.ReturnValue %0 : vector<4xi32>
}

// -----

func.func @expect_vector_bool(%val : vector<4xi1>, %expected : vector<4xi1>) -> vector<4xi1> {
  // CHECK: %{{.*}} = spirv.KHR.Expect %{{.*}}, %{{.*}} : vector<4xi1>
  %0 = spirv.KHR.Expect %val, %expected : vector<4xi1>
  spirv.ReturnValue %0 : vector<4xi1>
}

// -----

func.func @expect_type_mismatch(%val : i32, %expected : i64) -> i32 {
  // expected-error @+1 {{op failed to verify that all of {value, expectedValue, result} have same type}}
  %0 = "spirv.KHR.Expect"(%val, %expected) : (i32, i64) -> i32
  spirv.ReturnValue %0 : i32
}

// -----

func.func @expect_float_invalid(%val : f32, %expected : f32) -> f32 {
  // expected-error @+1 {{op operand #0 must be}}
  %0 = "spirv.KHR.Expect"(%val, %expected) : (f32, f32) -> f32
  spirv.ReturnValue %0 : f32
}
