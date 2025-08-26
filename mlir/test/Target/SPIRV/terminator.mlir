// RUN: mlir-translate -no-implicit-module -test-spirv-roundtrip %s | FileCheck %s

spirv.module Logical GLSL450 requires #spirv.vce<v1.0, [Shader], []> {
  // CHECK-LABEL: @ret
  spirv.func @ret() -> () "None" {
    // CHECK: spirv.Return
    spirv.Return
  }

  // CHECK-LABEL: @ret_val
  spirv.func @ret_val() -> (i32) "None" {
    %0 = spirv.Variable : !spirv.ptr<i32, Function>
    %1 = spirv.Load "Function" %0 : i32
    // CHECK: spirv.ReturnValue {{.*}} : i32
    spirv.ReturnValue %1 : i32
  }

  // CHECK-LABEL: @unreachable
  spirv.func @unreachable() "None" {
    spirv.Return
  // CHECK-NOT: ^bb
  ^bb1:
    // Unreachable blocks will be dropped during serialization.
    // CHECK-NOT: spirv.Unreachable
    spirv.Unreachable
  }

  // CHECK-LABEL: @kill
  spirv.func @kill() -> () "None" {
    // CHECK: spirv.Kill
    spirv.Kill
  }
}
