// RUN: mlir-translate -no-implicit-module -split-input-file -serialize-spirv -deserialize-spirv %s | FileCheck %s

spirv.module Logical GLSL450 requires #spirv.vce<v1.0, [Shader], []> {
  spirv.func @array_stride(%arg0 : !spirv.ptr<!spirv.array<4x!spirv.array<4xf32, stride=4>, stride=128>, StorageBuffer>, %arg1 : i32, %arg2 : i32) "None" {
    // CHECK: {{%.*}} = spirv.AccessChain {{%.*}}[{{%.*}}, {{%.*}}] : !spirv.ptr<!spirv.array<4 x !spirv.array<4 x f32, stride=4>, stride=128>, StorageBuffer>, i32, i32
    %2 = spirv.AccessChain %arg0[%arg1, %arg2] : !spirv.ptr<!spirv.array<4x!spirv.array<4xf32, stride=4>, stride=128>, StorageBuffer>, i32, i32 -> !spirv.ptr<f32, StorageBuffer>
    spirv.Return
  }
}

// -----

spirv.module Logical GLSL450 requires #spirv.vce<v1.0, [Shader], []> {
  // CHECK: spirv.GlobalVariable {{@.*}} : !spirv.ptr<!spirv.rtarray<f32, stride=4>, StorageBuffer>
  spirv.GlobalVariable @var0 : !spirv.ptr<!spirv.rtarray<f32, stride=4>, StorageBuffer>
  // CHECK: spirv.GlobalVariable {{@.*}} : !spirv.ptr<!spirv.rtarray<vector<4xf16>>, Input>
  spirv.GlobalVariable @var1 : !spirv.ptr<!spirv.rtarray<vector<4xf16>>, Input>
}
