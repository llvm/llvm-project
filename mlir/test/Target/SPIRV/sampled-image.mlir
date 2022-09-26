// RUN: mlir-translate -test-spirv-roundtrip %s | FileCheck %s

spirv.module Logical GLSL450 requires #spirv.vce<v1.0, [Shader], []> {
  // CHECK: !spirv.ptr<!spirv.sampled_image<!spirv.image<f32, Dim1D, NoDepth, NonArrayed, SingleSampled, NoSampler, Unknown>>, UniformConstant>
  spirv.GlobalVariable @var0 bind(0, 1) : !spirv.ptr<!spirv.sampled_image<!spirv.image<f32, Dim1D, NoDepth, NonArrayed, SingleSampled, NoSampler, Unknown>>, UniformConstant>

  // CHECK: !spirv.ptr<!spirv.sampled_image<!spirv.image<si32, SubpassData, DepthUnknown, Arrayed, MultiSampled, NoSampler, Unknown>>, UniformConstant>
  spirv.GlobalVariable @var1 bind(0, 0) : !spirv.ptr<!spirv.sampled_image<!spirv.image<si32, SubpassData, DepthUnknown, Arrayed, MultiSampled, NoSampler, Unknown>>, UniformConstant>

  // CHECK: !spirv.ptr<!spirv.sampled_image<!spirv.image<i32, Rect, DepthUnknown, Arrayed, MultiSampled, NeedSampler, R8ui>>, UniformConstant>
  spirv.GlobalVariable @var2 bind(0, 0) : !spirv.ptr<!spirv.sampled_image<!spirv.image<i32, Rect, DepthUnknown, Arrayed, MultiSampled, NeedSampler, R8ui>>, UniformConstant>
}
