// RUN: mlir-translate -no-implicit-module -test-spirv-roundtrip -split-input-file %s | FileCheck %s

spirv.module Logical GLSL450 requires #spirv.vce<v1.0, [Shader], []> {
  spirv.func @noop() -> () "None" {
    spirv.Return
  }
  // CHECK:      spirv.EntryPoint "GLCompute" @noop
  // CHECK-NEXT: spirv.ExecutionMode @noop "ContractionOff"
  spirv.EntryPoint "GLCompute" @noop
  spirv.ExecutionMode @noop "ContractionOff"
}

// -----

spirv.module Logical GLSL450 requires #spirv.vce<v1.0, [Shader], []> {
  // CHECK:       spirv.GlobalVariable @var2 : !spirv.ptr<f32, Input>
  // CHECK-NEXT:  spirv.GlobalVariable @var3 : !spirv.ptr<f32, Output>
  // CHECK-NEXT:  spirv.func @noop({{%.*}}: !spirv.ptr<f32, Input>, {{%.*}}: !spirv.ptr<f32, Output>) "None"
  // CHECK:       spirv.EntryPoint "GLCompute" @noop, @var2, @var3
  spirv.GlobalVariable @var2 : !spirv.ptr<f32, Input>
  spirv.GlobalVariable @var3 : !spirv.ptr<f32, Output>
  spirv.func @noop(%arg0 : !spirv.ptr<f32, Input>, %arg1 : !spirv.ptr<f32, Output>) -> () "None" {
    spirv.Return
  }
  spirv.EntryPoint "GLCompute" @noop, @var2, @var3
  spirv.ExecutionMode @noop "ContractionOff"
}
