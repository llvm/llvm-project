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
