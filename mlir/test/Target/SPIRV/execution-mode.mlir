// RUN: mlir-translate -test-spirv-roundtrip %s | FileCheck %s

spirv.module Logical GLSL450 requires #spirv.vce<v1.0, [Shader], []> {
  spirv.func @foo() -> () "None" {
    spirv.Return
  }
  spirv.EntryPoint "GLCompute" @foo
  // CHECK: spirv.ExecutionMode @foo "LocalSizeHint", 3, 4, 5
  spirv.ExecutionMode @foo "LocalSizeHint", 3, 4, 5
}
