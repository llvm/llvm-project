// RUN: mlir-translate -test-spirv-roundtrip %s | FileCheck %s

spv.module Logical GLSL450 requires #spv.vce<v1.0, [Shader], []> {
  spv.func @foo() -> () "None" {
    spv.Return
  }
  spv.EntryPoint "GLCompute" @foo
  // CHECK: spv.ExecutionMode @foo "LocalSizeHint", 3, 4, 5
  spv.ExecutionMode @foo "LocalSizeHint", 3, 4, 5
}
