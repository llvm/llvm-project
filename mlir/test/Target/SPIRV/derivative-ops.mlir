// RUN: mlir-translate -no-implicit-module -test-spirv-roundtrip %s | FileCheck %s

// RUN: %if spirv-tools %{ rm -rf %t %}
// RUN: %if spirv-tools %{ mkdir %t %}
// RUN: %if spirv-tools %{ mlir-translate --no-implicit-module --serialize-spirv --split-input-file --spirv-save-validation-files-with-prefix=%t/module %s %}
// RUN: %if spirv-tools %{ spirv-val %t %}

spirv.module Logical GLSL450 requires #spirv.vce<v1.0, [Shader, DerivativeControl], []> {
  // CHECK-LABEL: @derivatives
  spirv.func @derivatives() "None" {
    %s = spirv.Constant 1.0 : f32
    %v = spirv.Constant dense<1.0> : vector<4xf32>
    // CHECK: spirv.DPdxFine {{%.*}} : f32
    %0 = spirv.DPdxFine %s : f32
    // CHECK: spirv.DPdyFine {{%.*}} : vector<4xf32>
    %1 = spirv.DPdyFine %v : vector<4xf32>
    // CHECK: spirv.DPdxCoarse {{%.*}} : f32
    %2 = spirv.DPdxCoarse %s : f32
    // CHECK: spirv.DPdyCoarse {{%.*}} : vector<4xf32>
    %3 = spirv.DPdyCoarse %v : vector<4xf32>
    spirv.Return
  }
  spirv.EntryPoint "Fragment" @derivatives
  spirv.ExecutionMode @derivatives "OriginUpperLeft"
}
