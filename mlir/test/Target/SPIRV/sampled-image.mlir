// RUN: mlir-translate -test-spirv-roundtrip %s | FileCheck %s

spv.module Logical GLSL450 requires #spv.vce<v1.0, [Shader], []> {
  // CHECK: !spv.ptr<!spv.sampled_image<!spv.image<f32, Dim1D, NoDepth, NonArrayed, SingleSampled, NoSampler, Unknown>>, UniformConstant>
  spv.GlobalVariable @var0 bind(0, 1) : !spv.ptr<!spv.sampled_image<!spv.image<f32, Dim1D, NoDepth, NonArrayed, SingleSampled, NoSampler, Unknown>>, UniformConstant>

  // CHECK: !spv.ptr<!spv.sampled_image<!spv.image<si32, SubpassData, DepthUnknown, Arrayed, MultiSampled, NoSampler, Unknown>>, UniformConstant>
  spv.GlobalVariable @var1 bind(0, 0) : !spv.ptr<!spv.sampled_image<!spv.image<si32, SubpassData, DepthUnknown, Arrayed, MultiSampled, NoSampler, Unknown>>, UniformConstant>

  // CHECK: !spv.ptr<!spv.sampled_image<!spv.image<i32, Rect, DepthUnknown, Arrayed, MultiSampled, NeedSampler, R8ui>>, UniformConstant>
  spv.GlobalVariable @var2 bind(0, 0) : !spv.ptr<!spv.sampled_image<!spv.image<i32, Rect, DepthUnknown, Arrayed, MultiSampled, NeedSampler, R8ui>>, UniformConstant>
}
