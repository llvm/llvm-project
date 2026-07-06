// RUN: mlir-translate -no-implicit-module -split-input-file -test-spirv-roundtrip %s | FileCheck %s

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

// -----

// 2D + MultiSampled + NoSampler storage image — validates StorageImageMultisample.
spirv.module Logical GLSL450 requires #spirv.vce<v1.0, [Shader, Linkage, StorageImageMultisample], []> {
  // CHECK: !spirv.ptr<!spirv.image<f32, Dim2D, NoDepth, NonArrayed, MultiSampled, NoSampler, Unknown>, UniformConstant>
  spirv.GlobalVariable @img_2d_ms_storage bind(0, 0) :
    !spirv.ptr<!spirv.image<f32, Dim2D, NoDepth, NonArrayed, MultiSampled, NoSampler, Unknown>, UniformConstant>
}

// -----

// 2D + MultiSampled + Arrayed + NoSampler — validates StorageImageMultisample + ImageMSArray.
spirv.module Logical GLSL450 requires #spirv.vce<v1.0, [Shader, Linkage, StorageImageMultisample, ImageMSArray], []> {
  // CHECK: !spirv.ptr<!spirv.image<f32, Dim2D, NoDepth, Arrayed, MultiSampled, NoSampler, Unknown>, UniformConstant>
  spirv.GlobalVariable @img_2d_ms_arrayed bind(0, 0) :
    !spirv.ptr<!spirv.image<f32, Dim2D, NoDepth, Arrayed, MultiSampled, NoSampler, Unknown>, UniformConstant>
}

// -----

// Cube + Arrayed sampled image — validates ImageCubeArray.
spirv.module Logical GLSL450 requires #spirv.vce<v1.0, [Shader, Linkage, ImageCubeArray, SampledCubeArray], []> {
  // CHECK: !spirv.ptr<!spirv.image<f32, Cube, NoDepth, Arrayed, SingleSampled, NeedSampler, Unknown>, UniformConstant>
  spirv.GlobalVariable @img_cube_arrayed bind(0, 0) :
    !spirv.ptr<!spirv.image<f32, Cube, NoDepth, Arrayed, SingleSampled, NeedSampler, Unknown>, UniformConstant>
}

// -----

// 1D storage image — validates Image1D.
spirv.module Logical GLSL450 requires #spirv.vce<v1.0, [Shader, Linkage, Image1D, Sampled1D], []> {
  // CHECK: !spirv.ptr<!spirv.image<f32, Dim1D, NoDepth, NonArrayed, SingleSampled, NoSampler, Unknown>, UniformConstant>
  spirv.GlobalVariable @img_1d_storage bind(0, 0) :
    !spirv.ptr<!spirv.image<f32, Dim1D, NoDepth, NonArrayed, SingleSampled, NoSampler, Unknown>, UniformConstant>
}

// -----

// Buffer storage image — validates ImageBuffer.
spirv.module Logical GLSL450 requires #spirv.vce<v1.0, [Shader, Linkage, ImageBuffer, SampledBuffer], []> {
  // CHECK: !spirv.ptr<!spirv.image<f32, Buffer, NoDepth, NonArrayed, SingleSampled, NoSampler, Unknown>, UniformConstant>
  spirv.GlobalVariable @img_buffer_storage bind(0, 0) :
    !spirv.ptr<!spirv.image<f32, Buffer, NoDepth, NonArrayed, SingleSampled, NoSampler, Unknown>, UniformConstant>
}

// -----

// 64-bit integer sampled type — validates Int64ImageEXT + SPV_EXT_shader_image_int64.
spirv.module Logical GLSL450 requires #spirv.vce<v1.0, [Shader, Linkage, Int64, Int64ImageEXT], [SPV_EXT_shader_image_int64]> {
  // CHECK: !spirv.ptr<!spirv.image<i64, Dim2D, NoDepth, NonArrayed, SingleSampled, NoSampler, Unknown>, UniformConstant>
  spirv.GlobalVariable @img_2d_i64 bind(0, 0) :
    !spirv.ptr<!spirv.image<i64, Dim2D, NoDepth, NonArrayed, SingleSampled, NoSampler, Unknown>, UniformConstant>
}
