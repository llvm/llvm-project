// RUN: mlir-translate -no-implicit-module -test-spirv-roundtrip -split-input-file %s | FileCheck %s

// RUN: %if spirv-tools %{ rm -rf %t %}
// RUN: %if spirv-tools %{ mkdir %t %}
// RUN: %if spirv-tools %{ mlir-translate --no-implicit-module --serialize-spirv --split-input-file --spirv-save-validation-files-with-prefix=%t/module %s %}
// RUN: %if spirv-tools %{ spirv-val %t %}

spirv.module Logical OpenCL requires #spirv.vce<v1.0, [Kernel], []> {
  spirv.func @noop() -> () "None" {
    spirv.Return
  }
  // CHECK:      spirv.EntryPoint "Kernel" @noop
  // CHECK-NEXT: spirv.ExecutionMode @noop "ContractionOff"
  spirv.EntryPoint "Kernel" @noop
  spirv.ExecutionMode @noop "ContractionOff"
}

// -----

spirv.module Logical GLSL450 requires #spirv.vce<v1.0, [Shader], []> {
  // CHECK:       spirv.GlobalVariable @var2 : !spirv.ptr<f32, Input>
  // CHECK-NEXT:  spirv.GlobalVariable @var3 : !spirv.ptr<f32, Output>
  // CHECK-NEXT:  spirv.func @noop() "None"
  // CHECK:       spirv.EntryPoint "GLCompute" @noop, @var2, @var3
  spirv.GlobalVariable @var2 : !spirv.ptr<f32, Input>
  spirv.GlobalVariable @var3 : !spirv.ptr<f32, Output>
  spirv.func @noop() -> () "None" {
    spirv.Return
  }
  spirv.EntryPoint "GLCompute" @noop, @var2, @var3
  spirv.ExecutionMode @noop "LocalSize", 1, 1, 1
}
