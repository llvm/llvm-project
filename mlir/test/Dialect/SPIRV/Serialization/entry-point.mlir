// RUN: mlir-translate -test-spirv-roundtrip -split-input-file %s | FileCheck %s

spv.module Logical GLSL450 requires #spv.vce<v1.0, [Shader], []> {
  spv.func @noop() -> () "None" {
    spv.Return
  }
  // CHECK:      spv.EntryPoint "GLCompute" @noop
  // CHECK-NEXT: spv.ExecutionMode @noop "ContractionOff"
  spv.EntryPoint "GLCompute" @noop
  spv.ExecutionMode @noop "ContractionOff"
}

// -----

spv.module Logical GLSL450 requires #spv.vce<v1.0, [Shader], []> {
  // CHECK:       spv.globalVariable @var2 : !spv.ptr<f32, Input>
  // CHECK-NEXT:  spv.globalVariable @var3 : !spv.ptr<f32, Output>
  // CHECK-NEXT:  spv.func @noop({{%.*}}: !spv.ptr<f32, Input>, {{%.*}}: !spv.ptr<f32, Output>) "None"
  // CHECK:       spv.EntryPoint "GLCompute" @noop, @var2, @var3
  spv.globalVariable @var2 : !spv.ptr<f32, Input>
  spv.globalVariable @var3 : !spv.ptr<f32, Output>
  spv.func @noop(%arg0 : !spv.ptr<f32, Input>, %arg1 : !spv.ptr<f32, Output>) -> () "None" {
    spv.Return
  }
  spv.EntryPoint "GLCompute" @noop, @var2, @var3
  spv.ExecutionMode @noop "ContractionOff"
}
