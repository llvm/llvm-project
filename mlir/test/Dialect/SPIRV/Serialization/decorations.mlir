// RUN: mlir-translate -split-input-file -test-spirv-roundtrip %s | FileCheck %s

spv.module Logical GLSL450 requires #spv.vce<v1.0, [Shader], []> {
  // CHECK: location = 0 : i32
  spv.globalVariable @var1 {location = 0 : i32} : !spv.ptr<vector<4xf32>, Input>
}

// -----

spv.module Logical GLSL450 requires #spv.vce<v1.0, [Shader], []> {
  // CHECK: no_perspective
  spv.globalVariable @var1 {no_perspective} : !spv.ptr<vector<4xf32>, Input>
}

// -----

spv.module Logical GLSL450 requires #spv.vce<v1.0, [Shader], []> {
  // CHECK: flat
  spv.globalVariable @var2 {flat} : !spv.ptr<si32, Input>
}

