// RUN: mlir-translate -test-spirv-roundtrip %s | FileCheck %s

spv.module Logical GLSL450 requires #spv.vce<v1.0, [Shader], []> {
  // CHECK: !spv.ptr<!spv.image<f32, Dim1D, NoDepth, NonArrayed, SingleSampled, SamplerUnknown, Unknown>, UniformConstant>
  spv.GlobalVariable @var0 bind(0, 1) : !spv.ptr<!spv.image<f32, Dim1D, NoDepth, NonArrayed, SingleSampled, SamplerUnknown, Unknown>, UniformConstant>

  // CHECK: !spv.ptr<!spv.image<si32, Cube, IsDepth, NonArrayed, SingleSampled, NeedSampler, R8ui>, UniformConstant>
  spv.GlobalVariable @var1 : !spv.ptr<!spv.image<si32, Cube, IsDepth, NonArrayed, SingleSampled, NeedSampler, R8ui>, UniformConstant>

  // CHECK: !spv.ptr<!spv.image<i32, SubpassData, DepthUnknown, Arrayed, MultiSampled, NoSampler, Unknown>, UniformConstant>
  spv.GlobalVariable @var2 : !spv.ptr<!spv.image<i32, SubpassData, DepthUnknown, Arrayed, MultiSampled, NoSampler, Unknown>, UniformConstant>
}
