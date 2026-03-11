; RUN: llc -O0 -mtriple=spirv1.6-vulkan1.3-compute %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv1.6-vulkan1.3-compute %s -o - -filetype=obj | spirv-val --allow-offset-texture-operand --target-env vulkan1.3 %}

; CHECK-DAG: OpCapability Shader
; CHECK-DAG: OpCapability ImageGatherExtended

; CHECK-DAG: %[[float:[0-9]+]] = OpTypeFloat 32
; CHECK-DAG: %[[v4float:[0-9]+]] = OpTypeVector %[[float]] 4
; CHECK-DAG: %[[uint:[0-9]+]] = OpTypeInt 32 0
; CHECK-DAG: %[[v2uint:[0-9]+]] = OpTypeVector %[[uint]] 2
; CHECK-DAG: %[[v3uint:[0-9]+]] = OpTypeVector %[[uint]] 3
; CHECK-DAG: %[[image_1d:[0-9]+]] = OpTypeImage %[[float]] 1D 2 0 0 1 Unknown
; CHECK-DAG: %[[image_2d:[0-9]+]] = OpTypeImage %[[float]] 2D 2 0 0 1 Unknown
; CHECK-DAG: %[[image_3d:[0-9]+]] = OpTypeImage %[[float]] 3D 2 0 0 1 Unknown

; CHECK-DAG: %[[lod0:[0-9]+]] = OpConstant %[[uint]] 0
; CHECK-DAG: %[[uint_1:[0-9]+]] = OpConstant %[[uint]] 1
; CHECK-DAG: %[[coord_2d:[0-9]+]] = OpConstantNull %[[v2uint]]
; CHECK-DAG: %[[coord_3d:[0-9]+]] = OpConstantNull %[[v3uint]]
; CHECK-DAG: %[[offset_2d:[0-9]+]] = OpConstantComposite %[[v2uint]] %[[uint_1]] %[[uint_1]]
; CHECK-DAG: %[[offset_3d:[0-9]+]] = OpConstantComposite %[[v3uint]] %[[uint_1]] %[[uint_1]] %[[uint_1]]

@.str = private unnamed_addr constant [6 x i8] c"img1d\00", align 1
@.str.1 = private unnamed_addr constant [6 x i8] c"img2d\00", align 1
@.str.2 = private unnamed_addr constant [6 x i8] c"img3d\00", align 1

define void @test_load_1d() #0 {
entry:
  %img = tail call target("spirv.Image", float, 0, 2, 0, 0, 1, 0) @llvm.spv.resource.handlefrombinding.tspirv.Image_f32_0_2_0_0_1_0t(i32 0, i32 0, i32 1, i32 0, ptr @.str)
; CHECK: %[[img_val_1d:[0-9]+]] = OpLoad %[[image_1d]]
; CHECK: %[[res_1d:[0-9]+]] = OpImageFetch %[[v4float]] %[[img_val_1d]] %[[uint_1]] Lod %[[lod0]]
  %res = call <4 x float> @llvm.spv.resource.load.level.v4f32.tspirv.Image_f32_0_2_0_0_1_0t.i32.i32.i32(target("spirv.Image", float, 0, 2, 0, 0, 1, 0) %img, i32 1, i32 0, i32 0)
  ret void
}

define void @test_load_1d_offset() #0 {
entry:
  %img = tail call target("spirv.Image", float, 0, 2, 0, 0, 1, 0) @llvm.spv.resource.handlefrombinding.tspirv.Image_f32_0_2_0_0_1_0t(i32 0, i32 0, i32 1, i32 0, ptr @.str)
; CHECK: %[[img_val_1d_off:[0-9]+]] = OpLoad %[[image_1d]]
; CHECK: %[[res_1d_off:[0-9]+]] = OpImageFetch %[[v4float]] %[[img_val_1d_off]] %[[uint_1]] Lod|ConstOffset %[[lod0]] %[[uint_1]]
  %res = call <4 x float> @llvm.spv.resource.load.level.v4f32.tspirv.Image_f32_0_2_0_0_1_0t.i32.i32.i32(target("spirv.Image", float, 0, 2, 0, 0, 1, 0) %img, i32 1, i32 0, i32 1)
  ret void
}

define void @test_load_2d() #0 {
entry:
  %img = tail call target("spirv.Image", float, 1, 2, 0, 0, 1, 0) @llvm.spv.resource.handlefrombinding.tspirv.Image_f32_1_2_0_0_1_0t(i32 0, i32 1, i32 1, i32 0, ptr @.str.1)
; CHECK: %[[img_val_2d:[0-9]+]] = OpLoad %[[image_2d]]
; CHECK: %[[res_2d:[0-9]+]] = OpImageFetch %[[v4float]] %[[img_val_2d]] %[[coord_2d]] Lod %[[lod0]]
  %res = call <4 x float> @llvm.spv.resource.load.level.v4f32.tspirv.Image_f32_1_2_0_0_1_0t.v2i32.i32.v2i32(target("spirv.Image", float, 1, 2, 0, 0, 1, 0) %img, <2 x i32> zeroinitializer, i32 0, <2 x i32> zeroinitializer)
  ret void
}

define void @test_load_2d_offset() #0 {
entry:
  %img = tail call target("spirv.Image", float, 1, 2, 0, 0, 1, 0) @llvm.spv.resource.handlefrombinding.tspirv.Image_f32_1_2_0_0_1_0t(i32 0, i32 1, i32 1, i32 0, ptr @.str.1)
; CHECK: %[[img_val_2d_off:[0-9]+]] = OpLoad %[[image_2d]]
; CHECK: %[[res_2d_off:[0-9]+]] = OpImageFetch %[[v4float]] %[[img_val_2d_off]] %[[coord_2d]] Lod|ConstOffset %[[lod0]] %[[offset_2d]]
  %res = call <4 x float> @llvm.spv.resource.load.level.v4f32.tspirv.Image_f32_1_2_0_0_1_0t.v2i32.i32.v2i32(target("spirv.Image", float, 1, 2, 0, 0, 1, 0) %img, <2 x i32> zeroinitializer, i32 0, <2 x i32> <i32 1, i32 1>)
  ret void
}

define internal void @test_load_2d_non_const_offset(<2 x i32> %offset) {
entry:
; CHECK: %[[offset:[0-9]+]] = OpFunctionParameter %[[v2uint]]
  %img = tail call target("spirv.Image", float, 1, 2, 0, 0, 1, 0) @llvm.spv.resource.handlefrombinding.tspirv.Image_f32_1_2_0_0_1_0t(i32 0, i32 1, i32 1, i32 0, ptr @.str.1)
; CHECK: %[[img_val_2d_non_off:[0-9]+]] = OpLoad %[[image_2d]]
; CHECK: %[[res_2d_non_off:[0-9]+]] = OpImageFetch %[[v4float]] %[[img_val_2d_non_off]] %[[coord_2d]] Lod|Offset %[[lod0]] %[[offset]]
  %res = call <4 x float> @llvm.spv.resource.load.level.v4f32.tspirv.Image_f32_1_2_0_0_1_0t.v2i32.i32.v2i32(target("spirv.Image", float, 1, 2, 0, 0, 1, 0) %img, <2 x i32> zeroinitializer, i32 0, <2 x i32> %offset)
  ret void
}

define void @test_load_3d() #0 {
entry:
  %img = tail call target("spirv.Image", float, 2, 2, 0, 0, 1, 0) @llvm.spv.resource.handlefrombinding.tspirv.Image_f32_2_2_0_0_1_0t(i32 0, i32 2, i32 1, i32 0, ptr @.str.2)
; CHECK: %[[img_val_3d:[0-9]+]] = OpLoad %[[image_3d]]
; CHECK: %[[res_3d:[0-9]+]] = OpImageFetch %[[v4float]] %[[img_val_3d]] %[[coord_3d]] Lod %[[lod0]]
  %res = call <4 x float> @llvm.spv.resource.load.level.v4f32.tspirv.Image_f32_2_2_0_0_1_0t.v3i32.i32.v3i32(target("spirv.Image", float, 2, 2, 0, 0, 1, 0) %img, <3 x i32> zeroinitializer, i32 0, <3 x i32> zeroinitializer)
  ret void
}

define void @test_load_3d_offset() #0 {
entry:
  %img = tail call target("spirv.Image", float, 2, 2, 0, 0, 1, 0) @llvm.spv.resource.handlefrombinding.tspirv.Image_f32_2_2_0_0_1_0t(i32 0, i32 2, i32 1, i32 0, ptr @.str.2)
; CHECK: %[[img_val_3d_off:[0-9]+]] = OpLoad %[[image_3d]]
; CHECK: %[[res_3d_off:[0-9]+]] = OpImageFetch %[[v4float]] %[[img_val_3d_off]] %[[coord_3d]] Lod|ConstOffset %[[lod0]] %[[offset_3d]]
  %res = call <4 x float> @llvm.spv.resource.load.level.v4f32.tspirv.Image_f32_2_2_0_0_1_0t.v3i32.i32.v3i32(target("spirv.Image", float, 2, 2, 0, 0, 1, 0) %img, <3 x i32> zeroinitializer, i32 0, <3 x i32> <i32 1, i32 1, i32 1>)
  ret void
}

declare target("spirv.Image", float, 0, 2, 0, 0, 1, 0) @llvm.spv.resource.handlefrombinding.tspirv.Image_f32_0_2_0_0_1_0t(i32, i32, i32, i32, ptr)
declare <4 x float> @llvm.spv.resource.load.level.v4f32.tspirv.Image_f32_0_2_0_0_1_0t.i32.i32.i32(target("spirv.Image", float, 0, 2, 0, 0, 1, 0), i32, i32, i32)
declare target("spirv.Image", float, 1, 2, 0, 0, 1, 0) @llvm.spv.resource.handlefrombinding.tspirv.Image_f32_1_2_0_0_1_0t(i32, i32, i32, i32, ptr)
declare <4 x float> @llvm.spv.resource.load.level.v4f32.tspirv.Image_f32_1_2_0_0_1_0t.v2i32.i32.v2i32(target("spirv.Image", float, 1, 2, 0, 0, 1, 0), <2 x i32>, i32, <2 x i32>)
declare target("spirv.Image", float, 2, 2, 0, 0, 1, 0) @llvm.spv.resource.handlefrombinding.tspirv.Image_f32_2_2_0_0_1_0t(i32, i32, i32, i32, ptr)
declare <4 x float> @llvm.spv.resource.load.level.v4f32.tspirv.Image_f32_2_2_0_0_1_0t.v3i32.i32.v3i32(target("spirv.Image", float, 2, 2, 0, 0, 1, 0), <3 x i32>, i32, <3 x i32>)

attributes #0 = { "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" }
