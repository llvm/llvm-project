// RUN: mlir-opt -split-input-file -verify-diagnostics %s | FileCheck %s

//===----------------------------------------------------------------------===//
// spv.undef
//===----------------------------------------------------------------------===//

func @undef() -> () {
  // CHECK: %{{.*}} = spv.undef : f32
  %0 = spv.undef : f32
  // CHECK: %{{.*}} = spv.undef : vector<4xf32>
  %1 = spv.undef : vector<4xf32>
  spv.Return
}

// -----

func @undef() -> () {
  // expected-error @+2{{expected non-function type}}
  %0 = spv.undef :
  spv.Return
}

// -----

func @undef() -> () {
  // expected-error @+2{{expected ':'}}
  %0 = spv.undef
  spv.Return
}
