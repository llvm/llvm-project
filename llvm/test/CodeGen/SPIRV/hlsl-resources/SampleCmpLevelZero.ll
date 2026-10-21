; RUN: llc -O0 -mtriple=spirv-vulkan-compute %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-vulkan-compute %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: OpCapability Shader

; CHECK-DAG: %[[float:[0-9]+]] = OpTypeFloat 32
; CHECK-DAG: %[[image:[0-9]+]] = OpTypeImage %[[float]] 2D 2 0 0 1 Unknown
; CHECK-DAG: %[[sampled_image:[0-9]+]] = OpTypeSampledImage %[[image]]
; CHECK-DAG: %[[sampler:[0-9]+]] = OpTypeSampler
; CHECK-DAG: %[[v2float:[0-9]+]] = OpTypeVector %[[float]] 2
; CHECK-DAG: %[[v2int:[0-9]+]] = OpTypeVector %[[int:[0-9]+]] 2
; CHECK-DAG: %[[coord0:[0-9]+]] = OpConstantNull %[[v2float]]
; CHECK-DAG: %[[coord1_val:[0-9]+]] = OpConstant %[[float]] 0.5
; CHECK-DAG: %[[coord1:[0-9]+]] = OpConstantComposite %[[v2float]] %[[coord1_val]] %[[coord1_val]]
; CHECK-DAG: %[[zero:[0-9]+]] = OpConstantNull %[[float]]
; CHECK-DAG: %[[offset1_val:[0-9]+]] = OpConstant %[[int]] 1
; CHECK-DAG: %[[offset1:[0-9]+]] = OpConstantComposite %[[v2int]] %[[offset1_val]] %[[offset1_val]]

@.str = private unnamed_addr constant [4 x i8] c"img\00", align 1
@.str.1 = private unnamed_addr constant [5 x i8] c"samp\00", align 1
@.str.2 = private unnamed_addr constant [4 x i8] c"out\00", align 1

define void @main() {
entry:
  %img = tail call target("spirv.Image", float, 1, 2, 0, 0, 1, 0) @llvm.spv.resource.handlefrombinding.tspirv.Image_f32_1_2_0_0_1_0t(i32 0, i32 0, i32 1, i32 0, ptr @.str)
  %sampler = tail call target("spirv.Sampler") @llvm.spv.resource.handlefrombinding.tspirv.Samplert(i32 0, i32 1, i32 1, i32 0, ptr @.str.1)
  
; CHECK: %[[img_val:[0-9]+]] = OpLoad %[[image]]
; CHECK: %[[sampler_val:[0-9]+]] = OpLoad %[[sampler]]
; CHECK: %[[si:[0-9]+]] = OpSampledImage %[[sampled_image]] %[[img_val]] %[[sampler_val]]
; CHECK: %[[res0:[0-9]+]] = OpImageSampleDrefExplicitLod %[[float]] %[[si]] %[[coord0]] %[[zero]] Lod %[[zero]]{{[ ]*$}}
  %res0 = call float @llvm.spv.resource.samplecmplevelzero.f32.tspirv.Image_f32_1_2_0_0_1_0t.tspirv.Samplert.v2f32.v2i32(target("spirv.Image", float, 1, 2, 0, 0, 1, 0) %img, target("spirv.Sampler") %sampler, <2 x float> <float 0.0, float 0.0>, float 0.0, <2 x i32> zeroinitializer)

; CHECK: %[[img_val2:[0-9]+]] = OpLoad %[[image]]
; CHECK: %[[sampler_val2:[0-9]+]] = OpLoad %[[sampler]]
; CHECK: %[[si2:[0-9]+]] = OpSampledImage %[[sampled_image]] %[[img_val2]] %[[sampler_val2]]
; CHECK: %[[res1:[0-9]+]] = OpImageSampleDrefExplicitLod %[[float]] %[[si2]] %[[coord1]] %[[zero]] Lod|ConstOffset %[[zero]] %[[offset1]]{{[ ]*$}}
  %res1 = call float @llvm.spv.resource.samplecmplevelzero.f32.tspirv.Image_f32_1_2_0_0_1_0t.tspirv.Samplert.v2f32.v2i32(target("spirv.Image", float, 1, 2, 0, 0, 1, 0) %img, target("spirv.Sampler") %sampler, <2 x float> <float 0.5, float 0.5>, float 0.0, <2 x i32> <i32 1, i32 1>)

  %res = fadd float %res0, %res1

  %out = tail call target("spirv.Image", float, 5, 2, 0, 0, 2, 1) @llvm.spv.resource.handlefrombinding.tspirv.Image_f32_5_2_0_0_2_1t(i32 0, i32 2, i32 1, i32 0, ptr @.str.2)
  %out_ptr = call ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.Image_f32_5_2_0_0_2_1t(target("spirv.Image", float, 5, 2, 0, 0, 2, 1) %out, i32 0)
  store float %res, ptr addrspace(11) %out_ptr
  ret void
}

declare target("spirv.Image", float, 1, 2, 0, 0, 1, 0) @llvm.spv.resource.handlefrombinding.tspirv.Image_f32_1_2_0_0_1_0t(i32, i32, i32, i32, ptr)
declare target("spirv.Sampler") @llvm.spv.resource.handlefrombinding.tspirv.Samplert(i32, i32, i32, i32, ptr)
declare float @llvm.spv.resource.samplecmplevelzero.f32.tspirv.Image_f32_1_2_0_0_1_0t.tspirv.Samplert.v2f32.v2i32(target("spirv.Image", float, 1, 2, 0, 0, 1, 0), target("spirv.Sampler"), <2 x float>, float, <2 x i32>)
declare target("spirv.Image", float, 5, 2, 0, 0, 2, 1) @llvm.spv.resource.handlefrombinding.tspirv.Image_f32_5_2_0_0_2_1t(i32, i32, i32, i32, ptr)
declare ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.Image_f32_5_2_0_0_2_1t(target("spirv.Image", float, 5, 2, 0, 0, 2, 1) %out, i32)