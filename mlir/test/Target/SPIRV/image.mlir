// RUN: mlir-translate -no-implicit-module -test-spirv-roundtrip %s | FileCheck %s

// RUN: %if spirv-tools %{ rm -rf %t %}
// RUN: %if spirv-tools %{ mkdir %t %}
// RUN: %if spirv-tools %{ mlir-translate --no-implicit-module --serialize-spirv --split-input-file --spirv-save-validation-files-with-prefix=%t/module %s %}
// RUN: %if spirv-tools %{ spirv-val %t %}

spirv.module Logical GLSL450 requires #spirv.vce<v1.0, [Shader, Linkage, Sampled1D, StorageImageExtendedFormats, InputAttachment], []> {
  // CHECK: !spirv.ptr<!spirv.image<f32, Dim1D, NoDepth, NonArrayed, SingleSampled, SamplerUnknown, Unknown>, UniformConstant>
  spirv.GlobalVariable @var0 bind(0, 1) : !spirv.ptr<!spirv.image<f32, Dim1D, NoDepth, NonArrayed, SingleSampled, SamplerUnknown, Unknown>, UniformConstant>

  // CHECK: !spirv.ptr<!spirv.image<si32, Cube, IsDepth, NonArrayed, SingleSampled, NeedSampler, R8ui>, UniformConstant>
  spirv.GlobalVariable @var1 : !spirv.ptr<!spirv.image<si32, Cube, IsDepth, NonArrayed, SingleSampled, NeedSampler, R8ui>, UniformConstant>

  // CHECK: !spirv.ptr<!spirv.image<i32, SubpassData, DepthUnknown, Arrayed, MultiSampled, NoSampler, Unknown>, UniformConstant>
  spirv.GlobalVariable @var2 : !spirv.ptr<!spirv.image<i32, SubpassData, DepthUnknown, Arrayed, MultiSampled, NoSampler, Unknown>, UniformConstant>
}
