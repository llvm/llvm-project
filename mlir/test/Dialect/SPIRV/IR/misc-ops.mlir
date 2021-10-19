// RUN: mlir-opt -split-input-file -verify-diagnostics %s | FileCheck %s

//===----------------------------------------------------------------------===//
// spv.Undef
//===----------------------------------------------------------------------===//

func @undef() -> () {
  // CHECK: %{{.*}} = spv.Undef : f32
  %0 = spv.Undef : f32
  // CHECK: %{{.*}} = spv.Undef : vector<4xf32>
  %1 = spv.Undef : vector<4xf32>
  spv.Return
}

// -----

func @undef() -> () {
  // expected-error @+2{{expected non-function type}}
  %0 = spv.Undef :
  spv.Return
}

// -----

func @undef() -> () {
  // expected-error @+2{{expected ':'}}
  %0 = spv.Undef
  spv.Return
}
