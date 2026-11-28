// RUN: mlir-translate --no-implicit-module --test-spirv-roundtrip %s | FileCheck %s

// RUN: %if spirv-tools %{ rm -rf %t %}
// RUN: %if spirv-tools %{ mkdir %t %}
// RUN: %if spirv-tools %{ mlir-translate --no-implicit-module --serialize-spirv --spirv-save-validation-files-with-prefix=%t/module %s %}
// RUN: %if spirv-tools %{ spirv-val %t %}

spirv.module Logical GLSL450 requires #spirv.vce<v1.2, [Shader, Float64, FloatControls2], [SPV_KHR_float_controls2]> {
  spirv.SpecConstant @x = 3 : i32
  spirv.SpecConstant @y = 4 : i32
  spirv.SpecConstant @z = 5 : i32
  spirv.func @foo() -> () "None" {
    spirv.Return
  }
  spirv.EntryPoint "GLCompute" @foo
  // CHECK: spirv.ExecutionModeId @foo "LocalSizeId" @x, @y, @z
  spirv.ExecutionModeId @foo "LocalSizeId" @x, @y, @z
  // CHECK: spirv.ExecutionModeId @foo "FPFastMathDefault" f32, 0 : i32
  spirv.ExecutionModeId @foo "FPFastMathDefault" f32, 0 : i32
  // CHECK: spirv.ExecutionModeId @foo "FPFastMathDefault" f64, 458752 : i32
  spirv.ExecutionModeId @foo "FPFastMathDefault" f64, 458752 : i32
}
