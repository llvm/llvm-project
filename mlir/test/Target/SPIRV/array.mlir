// RUN: mlir-translate -no-implicit-module -split-input-file -test-spirv-roundtrip %s | FileCheck %s

// RUN: %if spirv-tools %{ rm -rf %t %}
// RUN: %if spirv-tools %{ mkdir %t %}
// RUN: %if spirv-tools %{ mlir-translate --no-implicit-module --serialize-spirv --split-input-file --spirv-save-validation-files-with-prefix=%t/module %s %}
// RUN: %if spirv-tools %{ spirv-val %t %}

spirv.module Logical GLSL450 requires #spirv.vce<v1.0, [Shader, Linkage], [SPV_KHR_storage_buffer_storage_class]> {
  spirv.func @array_stride(%arg0 : !spirv.ptr<!spirv.array<4x!spirv.array<4xf32, stride=4>, stride=128>, StorageBuffer>, %arg1 : i32, %arg2 : i32) "None" {
    // CHECK: {{%.*}} = spirv.AccessChain {{%.*}}[{{%.*}}, {{%.*}}] : !spirv.ptr<!spirv.array<4 x !spirv.array<4 x f32, stride=4>, stride=128>, StorageBuffer>, i32, i32
    %2 = spirv.AccessChain %arg0[%arg1, %arg2] : !spirv.ptr<!spirv.array<4x!spirv.array<4xf32, stride=4>, stride=128>, StorageBuffer>, i32, i32 -> !spirv.ptr<f32, StorageBuffer>
    spirv.Return
  }
}

// -----

spirv.module Logical GLSL450 requires #spirv.vce<v1.0, [Shader, Linkage, Float16], [SPV_KHR_storage_buffer_storage_class]> {
  // CHECK: spirv.GlobalVariable {{@.*}} : !spirv.ptr<!spirv.rtarray<f32, stride=4>, StorageBuffer>
  spirv.GlobalVariable @var0 : !spirv.ptr<!spirv.rtarray<f32, stride=4>, StorageBuffer>
  // CHECK: spirv.GlobalVariable {{@.*}} : !spirv.ptr<!spirv.rtarray<vector<4xf16>>, Input>
  spirv.GlobalVariable @var1 : !spirv.ptr<!spirv.rtarray<vector<4xf16>>, Input>
}
