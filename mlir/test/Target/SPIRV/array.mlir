// RUN: mlir-translate -split-input-file -test-spirv-roundtrip %s | FileCheck %s

spv.module Logical GLSL450 requires #spv.vce<v1.0, [Shader], []> {
  spv.func @array_stride(%arg0 : !spv.ptr<!spv.array<4x!spv.array<4xf32, stride=4>, stride=128>, StorageBuffer>, %arg1 : i32, %arg2 : i32) "None" {
    // CHECK: {{%.*}} = spv.AccessChain {{%.*}}[{{%.*}}, {{%.*}}] : !spv.ptr<!spv.array<4 x !spv.array<4 x f32, stride=4>, stride=128>, StorageBuffer>, i32, i32
    %2 = spv.AccessChain %arg0[%arg1, %arg2] : !spv.ptr<!spv.array<4x!spv.array<4xf32, stride=4>, stride=128>, StorageBuffer>, i32, i32
    spv.Return
  }
}

// -----

spv.module Logical GLSL450 requires #spv.vce<v1.0, [Shader], []> {
  // CHECK: spv.GlobalVariable {{@.*}} : !spv.ptr<!spv.rtarray<f32, stride=4>, StorageBuffer>
  spv.GlobalVariable @var0 : !spv.ptr<!spv.rtarray<f32, stride=4>, StorageBuffer>
  // CHECK: spv.GlobalVariable {{@.*}} : !spv.ptr<!spv.rtarray<vector<4xf16>>, Input>
  spv.GlobalVariable @var1 : !spv.ptr<!spv.rtarray<vector<4xf16>>, Input>
}
