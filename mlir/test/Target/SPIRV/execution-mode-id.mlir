// RUN: mlir-translate --no-implicit-module --test-spirv-roundtrip %s | FileCheck %s

// RUN: %if spirv-tools %{ rm -rf %t %}
// RUN: %if spirv-tools %{ mkdir %t %}
// RUN: %if spirv-tools %{ mlir-translate --no-implicit-module --serialize-spirv --spirv-save-validation-files-with-prefix=%t/module %s %}
// RUN: %if spirv-tools %{ spirv-val %t %}

spirv.module Logical GLSL450 requires #spirv.vce<v1.2, [Shader], []> {
  spirv.SpecConstant @x = 3 : i32
  spirv.SpecConstant @y = 4 : i32
  spirv.SpecConstant @z = 5 : i32
  spirv.func @foo() -> () "None" {
    spirv.Return
  }
  spirv.EntryPoint "GLCompute" @foo
  // CHECK: spirv.ExecutionModeId @foo "LocalSizeId" @x, @y, @z
  spirv.ExecutionModeId @foo "LocalSizeId" @x, @y, @z
}
