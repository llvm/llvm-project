; RUN: llc -O0 -mtriple=spirv-vulkan1.3-compute %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-vulkan1.3-compute %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: OpCapability Shader

; CHECK-DAG: %[[float:[0-9]+]] = OpTypeFloat 32
; CHECK-DAG: %[[v4float:[0-9]+]] = OpTypeVector %[[float]] 4
; CHECK-DAG: %[[image:[0-9]+]] = OpTypeImage %[[float]] 2D 2 1 0 1 Unknown
; CHECK-DAG: %[[sampled_image:[0-9]+]] = OpTypeSampledImage %[[image]]
; CHECK-DAG: %[[sampler:[0-9]+]] = OpTypeSampler
; CHECK-DAG: %[[v3float:[0-9]+]] = OpTypeVector %[[float]] 3
; CHECK-DAG: %[[int:[0-9]+]] = OpTypeInt 32 0
; CHECK-DAG: %[[v2int:[0-9]+]] = OpTypeVector %[[int]] 2
; CHECK-DAG: %[[coord:[0-9]+]] = OpConstantComposite %[[v3float]]
; CHECK-DAG: %[[component0:[0-9]+]] = OpConstant %[[int]] 0
; CHECK-DAG: %[[const1:[0-9]+]] = OpConstant %[[int]] 1
; CHECK-DAG: %[[offset:[0-9]+]] = OpConstantComposite %[[v2int]] %[[const1]] %[[const1]]

@.str = private unnamed_addr constant [4 x i8] c"img\00", align 1
@.str.1 = private unnamed_addr constant [5 x i8] c"samp\00", align 1

define void @main() {
entry:
  %img = tail call target("spirv.Image", float, 1, 2, 1, 0, 1, 0) @llvm.spv.resource.handlefrombinding.tspirv.Image_f32_1_2_1_0_1_0t(i32 0, i32 0, i32 1, i32 0, ptr @.str)
  %sampler = tail call target("spirv.Sampler") @llvm.spv.resource.handlefrombinding.tspirv.Samplert(i32 0, i32 1, i32 1, i32 0, ptr @.str.1)

; CHECK-DAG: %[[img_val:[0-9]+]] = OpLoad %[[image]] %[[image_var:[0-9]+]]
; CHECK-DAG: %[[sampler_val:[0-9]+]] = OpLoad %[[sampler]] %[[sampler_var:[0-9]+]]
; CHECK: %[[si:[0-9]+]] = OpSampledImage %[[sampled_image]] %[[img_val]] %[[sampler_val]]
; CHECK: %[[res0:[0-9]+]] = OpImageGather %[[v4float]] %[[si]] %[[coord]] %[[component0]]
  %res0 = call <4 x float> @llvm.spv.resource.gather.v4f32.tspirv.Image_f32_1_2_1_0_1_0t.tspirv.Samplert.v3f32.i32.v2i32(target("spirv.Image", float, 1, 2, 1, 0, 1, 0) %img, target("spirv.Sampler") %sampler, <3 x float> zeroinitializer, i32 0, <2 x i32> zeroinitializer)

; CHECK: %[[img_val2:[0-9]+]] = OpLoad %[[image]] %[[image_var]]
; CHECK: %[[sampler_val2:[0-9]+]] = OpLoad %[[sampler]] %[[sampler_var]]
; CHECK: %[[si2:[0-9]+]] = OpSampledImage %[[sampled_image]] %[[img_val2]] %[[sampler_val2]]
; CHECK: %[[res1:[0-9]+]] = OpImageGather %[[v4float]] %[[si2]] %[[coord]] %[[const1]] ConstOffset %[[offset]]
  %res1 = call <4 x float> @llvm.spv.resource.gather.v4f32.tspirv.Image_f32_1_2_1_0_1_0t.tspirv.Samplert.v3f32.i32.v2i32(target("spirv.Image", float, 1, 2, 1, 0, 1, 0) %img, target("spirv.Sampler") %sampler, <3 x float> zeroinitializer, i32 1, <2 x i32> <i32 1, i32 1>)

  ret void
}

declare target("spirv.Image", float, 1, 2, 1, 0, 1, 0) @llvm.spv.resource.handlefrombinding.tspirv.Image_f32_1_2_1_0_1_0t(i32, i32, i32, i32, ptr)
declare target("spirv.Sampler") @llvm.spv.resource.handlefrombinding.tspirv.Samplert(i32, i32, i32, i32, ptr)
declare <4 x float> @llvm.spv.resource.gather.v4f32.tspirv.Image_f32_1_2_1_0_1_0t.tspirv.Samplert.v3f32.i32.v2i32(target("spirv.Image", float, 1, 2, 1, 0, 1, 0), target("spirv.Sampler"), <3 x float>, i32, <2 x i32>)
