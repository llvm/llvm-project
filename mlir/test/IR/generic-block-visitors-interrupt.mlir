// RUN: mlir-opt -test-generic-ir-block-visitors-interrupt -allow-unregistered-dialect -split-input-file %s | FileCheck %s

func.func @main(%arg0: f32) -> f32 {
  %v1 = "foo"() {interrupt = true} : () -> f32
  %v2 = arith.addf %v1, %arg0 : f32
  return %v2 : f32
}

// CHECK: step 0 walk was interrupted
