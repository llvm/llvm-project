// RUN: mlir-translate -split-input-file -test-spirv-roundtrip %s | FileCheck %s

spv.module Logical GLSL450 requires #spv.vce<v1.0, [Shader], []> {
  spv.func @matrix_type(%arg0 : !spv.ptr<!spv.matrix<3 x vector<3xf32>>, StorageBuffer>, %arg1 : i32) "None" {
    // CHECK: {{%.*}} = spv.AccessChain {{%.*}}[{{%.*}}] : !spv.ptr<!spv.matrix<3 x vector<3xf32>>, StorageBuffer>
    %2 = spv.AccessChain %arg0[%arg1] : !spv.ptr<!spv.matrix<3 x vector<3xf32>>, StorageBuffer>
    spv.Return
  }
}

// -----

spv.module Logical GLSL450 requires #spv.vce<v1.0, [Shader], []> {
  // CHECK: spv.globalVariable {{@.*}} : !spv.ptr<!spv.matrix<3 x vector<3xf32>>, StorageBuffer>
  spv.globalVariable @var0 : !spv.ptr<!spv.matrix<3 x vector<3xf32>>, StorageBuffer>

  // CHECK: spv.globalVariable {{@.*}} : !spv.ptr<!spv.matrix<2 x vector<3xf32>>, StorageBuffer>
  spv.globalVariable @var1 : !spv.ptr<!spv.matrix<2 x vector<3xf32>>, StorageBuffer>

  // CHECK: spv.globalVariable {{@.*}} : !spv.ptr<!spv.matrix<4 x vector<4xf16>>, StorageBuffer>
  spv.globalVariable @var2 : !spv.ptr<!spv.matrix<4 x vector<4xf16>>, StorageBuffer>
}
