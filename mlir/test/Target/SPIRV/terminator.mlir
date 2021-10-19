// RUN: mlir-translate -test-spirv-roundtrip %s | FileCheck %s

spv.module Logical GLSL450 requires #spv.vce<v1.0, [Shader], []> {
  // CHECK-LABEL: @ret
  spv.func @ret() -> () "None" {
    // CHECK: spv.Return
    spv.Return
  }

  // CHECK-LABEL: @ret_val
  spv.func @ret_val() -> (i32) "None" {
    %0 = spv.Variable : !spv.ptr<i32, Function>
    %1 = spv.Load "Function" %0 : i32
    // CHECK: spv.ReturnValue {{.*}} : i32
    spv.ReturnValue %1 : i32
  }

  // CHECK-LABEL: @unreachable
  spv.func @unreachable() "None" {
    spv.Return
  // CHECK-NOT: ^bb
  ^bb1:
    // Unreachable blocks will be dropped during serialization.
    // CHECK-NOT: spv.Unreachable
    spv.Unreachable
  }
}
