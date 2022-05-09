// RUN: mlir-opt -split-input-file -verify-diagnostics %s | FileCheck %s

//===----------------------------------------------------------------------===//
// spv.Undef
//===----------------------------------------------------------------------===//

func.func @undef() -> () {
  // CHECK: %{{.*}} = spv.Undef : f32
  %0 = spv.Undef : f32
  // CHECK: %{{.*}} = spv.Undef : vector<4xf32>
  %1 = spv.Undef : vector<4xf32>
  spv.Return
}

// -----

func.func @undef() -> () {
  // expected-error @+2{{expected non-function type}}
  %0 = spv.Undef :
  spv.Return
}

// -----

func.func @undef() -> () {
  // expected-error @+2{{expected ':'}}
  %0 = spv.Undef
  spv.Return
}

// -----

func.func @assume_true(%arg : i1) -> () {
  // CHECK: spv.AssumeTrueKHR %{{.*}}
  spv.AssumeTrueKHR %arg
  spv.Return
}

// -----

func.func @assume_true(%arg : f32) -> () {
  // expected-error @+2{{use of value '%arg' expects different type than prior uses: 'i1' vs 'f32'}}
  // expected-note @-2 {{prior use here}}
  spv.AssumeTrueKHR %arg
  spv.Return
}
