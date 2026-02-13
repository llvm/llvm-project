// RUN: mlir-translate -no-implicit-module -test-spirv-roundtrip %s | FileCheck %s

// RUN: %if spirv-tools %{ rm -rf %t %}
// RUN: %if spirv-tools %{ mkdir %t %}
// RUN: %if spirv-tools %{ mlir-translate --no-implicit-module --serialize-spirv --split-input-file --spirv-save-validation-files-with-prefix=%t/module %s %}
// RUN: %if spirv-tools %{ spirv-val %t %}

spirv.module Logical GLSL450 requires #spirv.vce<v1.0, [Shader, Sampled1D, SampledRect, StorageImageExtendedFormats, Linkage], []> {
  // CHECK: !spirv.ptr<!spirv.sampled_image<!spirv.image<f32, Dim1D, NoDepth, NonArrayed, SingleSampled, NeedSampler, Unknown>>, UniformConstant>
  spirv.GlobalVariable @var0 bind(0, 1) : !spirv.ptr<!spirv.sampled_image<!spirv.image<f32, Dim1D, NoDepth, NonArrayed, SingleSampled, NeedSampler, Unknown>>, UniformConstant>

  // CHECK: !spirv.ptr<!spirv.sampled_image<!spirv.image<si32, Cube, DepthUnknown, Arrayed, MultiSampled, NeedSampler, Unknown>>, UniformConstant>
  spirv.GlobalVariable @var1 bind(0, 0) : !spirv.ptr<!spirv.sampled_image<!spirv.image<si32, Cube, DepthUnknown, Arrayed, MultiSampled, NeedSampler, Unknown>>, UniformConstant>

  // CHECK: !spirv.ptr<!spirv.sampled_image<!spirv.image<i32, Rect, DepthUnknown, Arrayed, MultiSampled, NeedSampler, R8ui>>, UniformConstant>
  spirv.GlobalVariable @var2 bind(0, 0) : !spirv.ptr<!spirv.sampled_image<!spirv.image<i32, Rect, DepthUnknown, Arrayed, MultiSampled, NeedSampler, R8ui>>, UniformConstant>
}
