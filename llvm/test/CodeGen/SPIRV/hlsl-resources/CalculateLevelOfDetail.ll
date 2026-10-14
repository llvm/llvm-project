; RUN: llc -O0 -mtriple=spirv1.6-vulkan1.3-library %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv1.6-vulkan1.3-library %s -o - -filetype=obj | spirv-val --target-env vulkan1.3 %}

; CHECK-DAG: %[[float:[0-9]+]] = OpTypeFloat 32
; CHECK-DAG: %[[image:[0-9]+]] = OpTypeImage %[[float]] 2D 2 0 0 1 Unknown
; CHECK-DAG: %[[sampled_image:[0-9]+]] = OpTypeSampledImage %[[image]]
; CHECK-DAG: %[[sampler:[0-9]+]] = OpTypeSampler
; CHECK-DAG: %[[v2float:[0-9]+]] = OpTypeVector %[[float]] 2
; CHECK-DAG: %[[coord0:[0-9]+]] = OpConstantNull %[[v2float]]

@.str = private unnamed_addr constant [4 x i8] c"img\00", align 1
@.str.1 = private unnamed_addr constant [5 x i8] c"samp\00", align 1

define void @main() #0 {
entry:
  %img = tail call target("spirv.Image", float, 1, 2, 0, 0, 1, 0) @llvm.spv.resource.handlefrombinding.tspirv.Image_f32_1_2_0_0_1_0t(i32 0, i32 0, i32 1, i32 0, ptr @.str)
  %sampler = tail call target("spirv.Sampler") @llvm.spv.resource.handlefrombinding.tspirv.Samplert(i32 0, i32 1, i32 1, i32 0, ptr @.str.1)
  
; CHECK: %[[img_val:[0-9]+]] = OpLoad %[[image]]
; CHECK: %[[sampler_val:[0-9]+]] = OpLoad %[[sampler]]
; CHECK: %[[si:[0-9]+]] = OpSampledImage %[[sampled_image]] %[[img_val]] %[[sampler_val]]
; CHECK: %[[res_vec:[0-9]+]] = OpImageQueryLod %[[v2float]] %[[si]] %[[coord0]]
; CHECK: %[[res0:[0-9]+]] = OpCompositeExtract %[[float]] %[[res_vec]] 0
  %res0 = call float @llvm.spv.resource.calculate.lod.f32.tspirv.Image_f32_1_2_0_0_1_0t.tspirv.Samplert.v2f32(target("spirv.Image", float, 1, 2, 0, 0, 1, 0) %img, target("spirv.Sampler") %sampler, <2 x float> <float 0.0, float 0.0>)

; CHECK: %[[img_val2:[0-9]+]] = OpLoad %[[image]]
; CHECK: %[[sampler_val2:[0-9]+]] = OpLoad %[[sampler]]
; CHECK: %[[si2:[0-9]+]] = OpSampledImage %[[sampled_image]] %[[img_val2]] %[[sampler_val2]]
; CHECK: %[[res_vec2:[0-9]+]] = OpImageQueryLod %[[v2float]] %[[si2]] %[[coord0]]
; CHECK: %[[res1:[0-9]+]] = OpCompositeExtract %[[float]] %[[res_vec2]] 1
  %res1 = call float @llvm.spv.resource.calculate.lod.unclamped.f32.tspirv.Image_f32_1_2_0_0_1_0t.tspirv.Samplert.v2f32(target("spirv.Image", float, 1, 2, 0, 0, 1, 0) %img, target("spirv.Sampler") %sampler, <2 x float> <float 0.0, float 0.0>)

  ret void
}

attributes #0 = { "hlsl.shader"="pixel" }
