; RUN: llc -O0 -mtriple=spirv-vulkan1.3-compute %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-vulkan1.3-compute %s -o - -filetype=obj | spirv-val %}

; CHECK: OpCapability ImageGatherExtended

; CHECK-DAG: %[[float:[0-9]+]] = OpTypeFloat 32
; CHECK-DAG: %[[v4float:[0-9]+]] = OpTypeVector %[[float]] 4
; CHECK-DAG: %[[image:[0-9]+]] = OpTypeImage %[[float]] 2D 0 0 0 1 Unknown
; CHECK-DAG: %[[sampled_image:[0-9]+]] = OpTypeSampledImage %[[image]]
; CHECK-DAG: %[[sampler:[0-9]+]] = OpTypeSampler
; CHECK-DAG: %[[v2float:[0-9]+]] = OpTypeVector %[[float]] 2
; CHECK-DAG: %[[int:[0-9]+]] = OpTypeInt 32 0
; CHECK-DAG: %[[v2int:[0-9]+]] = OpTypeVector %[[int]] 2
; CHECK-DAG: %[[coord:[0-9]+]] = OpConstantNull %[[v2float]]
; CHECK-DAG: %[[component0:[0-9]+]] = OpConstant %[[int]] 0
; CHECK-DAG: %[[const1:[0-9]+]] = OpConstant %[[int]] 1
; CHECK-DAG: %[[compare:[0-9]+]] = OpConstant %[[float]] 0.5
; CHECK-DAG: %[[offset:[0-9]+]] = OpConstantComposite %[[v2int]] %[[const1]] %[[const1]]
; CHECK-DAG: %[[image_cube:[0-9]+]] = OpTypeImage %[[float]] Cube 0 0 0 1 Unknown
; CHECK-DAG: %[[sampled_image_cube:[0-9]+]] = OpTypeSampledImage %[[image_cube]]
; CHECK-DAG: %[[v3float:[0-9]+]] = OpTypeVector %[[float]] 3
; CHECK-DAG: %[[coord_cube:[0-9]+]] = OpConstantNull %[[v3float]]
; CHECK-DAG: %[[ptr_int:[0-9]+]] = OpTypePointer Function %[[v2int]]

@.str = private unnamed_addr constant [4 x i8] c"img\00", align 1
@.str.1 = private unnamed_addr constant [5 x i8] c"samp\00", align 1
@.str.cube = private unnamed_addr constant [9 x i8] c"img_cube\00", align 1

@offset_var = private global <2 x i32> zeroinitializer

define void @main() {
entry:
; CHECK-DAG: %[[offset_var:[0-9]+]] = OpVariable %[[ptr_int]] Function

  %img = tail call target("spirv.Image", float, 1, 0, 0, 0, 1, 0) @llvm.spv.resource.handlefrombinding.tspirv.Image_f32_1_0_0_0_1_0t(i32 0, i32 0, i32 1, i32 0, ptr @.str)
  %sampler = tail call target("spirv.Sampler") @llvm.spv.resource.handlefrombinding.tspirv.Samplert(i32 0, i32 1, i32 1, i32 0, ptr @.str.1)

; CHECK: %[[img_val:[0-9]+]] = OpLoad %[[image]]
; CHECK: %[[sampler_val:[0-9]+]] = OpLoad %[[sampler]]
; CHECK: %[[si:[0-9]+]] = OpSampledImage %[[sampled_image]] %[[img_val]] %[[sampler_val]]
; CHECK: %[[res0:[0-9]+]] = OpImageGather %[[v4float]] %[[si]] %[[coord]] %[[component0]]{{[ ]*$}}
  %res0 = call <4 x float> @llvm.spv.resource.gather.v4f32.tspirv.Image_f32_1_0_0_0_1_0t.tspirv.Samplert.v2f32.i32.v2i32(target("spirv.Image", float, 1, 0, 0, 0, 1, 0) %img, target("spirv.Sampler") %sampler, <2 x float> zeroinitializer, i32 0, <2 x i32> zeroinitializer)

; CHECK: %[[img_val2:[0-9]+]] = OpLoad %[[image]]
; CHECK: %[[sampler_val2:[0-9]+]] = OpLoad %[[sampler]]
; CHECK: %[[si2:[0-9]+]] = OpSampledImage %[[sampled_image]] %[[img_val2]] %[[sampler_val2]]
; CHECK: %[[res1:[0-9]+]] = OpImageGather %[[v4float]] %[[si2]] %[[coord]] %[[const1]] ConstOffset %[[offset]]{{[ ]*$}}
  %res1 = call <4 x float> @llvm.spv.resource.gather.v4f32.tspirv.Image_f32_1_0_0_0_1_0t.tspirv.Samplert.v2f32.i32.v2i32(target("spirv.Image", float, 1, 0, 0, 0, 1, 0) %img, target("spirv.Sampler") %sampler, <2 x float> zeroinitializer, i32 1, <2 x i32> <i32 1, i32 1>)

; CHECK: %[[img_val3:[0-9]+]] = OpLoad %[[image]]
; CHECK: %[[sampler_val3:[0-9]+]] = OpLoad %[[sampler]]
; CHECK: %[[si3:[0-9]+]] = OpSampledImage %[[sampled_image]] %[[img_val3]] %[[sampler_val3]]
; CHECK: %[[res2:[0-9]+]] = OpImageDrefGather %[[v4float]] %[[si3]] %[[coord]] %[[compare]]{{[ ]*$}}
  %res2 = call <4 x float> @llvm.spv.resource.gather.cmp.v4f32.tspirv.Image_f32_1_0_0_0_1_0t.tspirv.Samplert.v2f32.f32.v2i32(target("spirv.Image", float, 1, 0, 0, 0, 1, 0) %img, target("spirv.Sampler") %sampler, <2 x float> zeroinitializer, float 0.5, <2 x i32> zeroinitializer)

; CHECK: %[[img_val4:[0-9]+]] = OpLoad %[[image]]
; CHECK: %[[sampler_val4:[0-9]+]] = OpLoad %[[sampler]]
; CHECK: %[[si4:[0-9]+]] = OpSampledImage %[[sampled_image]] %[[img_val4]] %[[sampler_val4]]
; CHECK: %[[res3:[0-9]+]] = OpImageDrefGather %[[v4float]] %[[si4]] %[[coord]] %[[compare]] ConstOffset %[[offset]]{{[ ]*$}}
  %res3 = call <4 x float> @llvm.spv.resource.gather.cmp.v4f32.tspirv.Image_f32_1_0_0_0_1_0t.tspirv.Samplert.v2f32.f32.v2i32(target("spirv.Image", float, 1, 0, 0, 0, 1, 0) %img, target("spirv.Sampler") %sampler, <2 x float> zeroinitializer, float 0.5, <2 x i32> <i32 1, i32 1>)

; CHECK: %[[off_dyn:[0-9]+]] = OpLoad %[[v2int]] %[[offset_var]]
; CHECK: %[[img_val5:[0-9]+]] = OpLoad %[[image]]
; CHECK: %[[sampler_val5:[0-9]+]] = OpLoad %[[sampler]]
; CHECK: %[[si5:[0-9]+]] = OpSampledImage %[[sampled_image]] %[[img_val5]] %[[sampler_val5]]
; CHECK: %[[res4:[0-9]+]] = OpImageGather %[[v4float]] %[[si5]] %[[coord]] %[[component0]] Offset %[[off_dyn]]{{[ ]*$}}
  %off_dyn = load <2 x i32>, ptr @offset_var
  %res4 = call <4 x float> @llvm.spv.resource.gather.v4f32.tspirv.Image_f32_1_0_0_0_1_0t.tspirv.Samplert.v2f32.i32.v2i32(target("spirv.Image", float, 1, 0, 0, 0, 1, 0) %img, target("spirv.Sampler") %sampler, <2 x float> zeroinitializer, i32 0, <2 x i32> %off_dyn)

  ret void
}

define void @main_cube() {
entry:
  %img = tail call target("spirv.Image", float, 3, 0, 0, 0, 1, 0) @llvm.spv.resource.handlefrombinding.tspirv.Image_f32_3_0_0_0_1_0t(i32 0, i32 2, i32 1, i32 0, ptr @.str.cube)
  %sampler = tail call target("spirv.Sampler") @llvm.spv.resource.handlefrombinding.tspirv.Samplert(i32 0, i32 3, i32 1, i32 0, ptr @.str.1)
  
; CHECK: %[[img_val_cube:[0-9]+]] = OpLoad %[[image_cube]]
; CHECK: %[[sampler_val_cube:[0-9]+]] = OpLoad %[[sampler]]
; CHECK: %[[si_cube:[0-9]+]] = OpSampledImage %[[sampled_image_cube]] %[[img_val_cube]] %[[sampler_val_cube]]
; CHECK: %[[res0_cube:[0-9]+]] = OpImageGather %[[v4float]] %[[si_cube]] %[[coord_cube]] %[[component0]]{{[ ]*$}}
  %res0 = call <4 x float> @llvm.spv.resource.gather.v4f32.tspirv.Image_f32_3_0_0_0_1_0t.tspirv.Samplert.v3f32.i32.v2i32(target("spirv.Image", float, 3, 0, 0, 0, 1, 0) %img, target("spirv.Sampler") %sampler, <3 x float> zeroinitializer, i32 0, <2 x i32> zeroinitializer)

; CHECK: %[[img_val3_cube:[0-9]+]] = OpLoad %[[image_cube]]
; CHECK: %[[sampler_val3_cube:[0-9]+]] = OpLoad %[[sampler]]
; CHECK: %[[si3_cube:[0-9]+]] = OpSampledImage %[[sampled_image_cube]] %[[img_val3_cube]] %[[sampler_val3_cube]]
; CHECK: %[[res2_cube:[0-9]+]] = OpImageDrefGather %[[v4float]] %[[si3_cube]] %[[coord_cube]] %[[compare]]{{[ ]*$}}
  %res2 = call <4 x float> @llvm.spv.resource.gather.cmp.v4f32.tspirv.Image_f32_3_0_0_0_1_0t.tspirv.Samplert.v3f32.f32.v2i32(target("spirv.Image", float, 3, 0, 0, 0, 1, 0) %img, target("spirv.Sampler") %sampler, <3 x float> zeroinitializer, float 0.5, <2 x i32> zeroinitializer)

  ret void
}

declare target("spirv.Image", float, 1, 0, 0, 0, 1, 0) @llvm.spv.resource.handlefrombinding.tspirv.Image_f32_1_0_0_0_1_0t(i32, i32, i32, i32, ptr)
declare target("spirv.Sampler") @llvm.spv.resource.handlefrombinding.tspirv.Samplert(i32, i32, i32, i32, ptr)
declare <4 x float> @llvm.spv.resource.gather.v4f32.tspirv.Image_f32_1_0_0_0_1_0t.tspirv.Samplert.v2f32.i32.v2i32(target("spirv.Image", float, 1, 0, 0, 0, 1, 0), target("spirv.Sampler"), <2 x float>, i32, <2 x i32>)
declare <4 x float> @llvm.spv.resource.gather.cmp.v4f32.tspirv.Image_f32_1_0_0_0_1_0t.tspirv.Samplert.v2f32.f32.v2i32(target("spirv.Image", float, 1, 0, 0, 0, 1, 0), target("spirv.Sampler"), <2 x float>, float, <2 x i32>)

declare target("spirv.Image", float, 3, 0, 0, 0, 1, 0) @llvm.spv.resource.handlefrombinding.tspirv.Image_f32_3_0_0_0_1_0t(i32, i32, i32, i32, ptr)
declare <4 x float> @llvm.spv.resource.gather.v4f32.tspirv.Image_f32_3_0_0_0_1_0t.tspirv.Samplert.v3f32.i32.v2i32(target("spirv.Image", float, 3, 0, 0, 0, 1, 0), target("spirv.Sampler"), <3 x float>, i32, <2 x i32>)
declare <4 x float> @llvm.spv.resource.gather.cmp.v4f32.tspirv.Image_f32_3_0_0_0_1_0t.tspirv.Samplert.v3f32.f32.v2i32(target("spirv.Image", float, 3, 0, 0, 0, 1, 0), target("spirv.Sampler"), <3 x float>, float, <2 x i32>)
