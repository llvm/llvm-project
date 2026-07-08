; RUN: llc -O0 -mtriple=spirv1.6-vulkan1.3-compute %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv1.6-vulkan1.3-compute %s -o - -filetype=obj | spirv-val --allow-offset-texture-operand --target-env vulkan1.3 %}

; CHECK-DAG: OpCapability Shader

; CHECK-DAG: %[[float:[0-9]+]] = OpTypeFloat 32
; CHECK-DAG: %[[v4float:[0-9]+]] = OpTypeVector %[[float]] 4
; CHECK-DAG: %[[uint:[0-9]+]] = OpTypeInt 32 0
; CHECK-DAG: %[[v2uint:[0-9]+]] = OpTypeVector %[[uint]] 2
; CHECK-DAG: %[[v3uint:[0-9]+]] = OpTypeVector %[[uint]] 3
; CHECK-DAG: %[[image_2darray:[0-9]+]] = OpTypeImage %[[float]] 2D 2 1 0 1 Unknown

; CHECK-DAG: %[[lod0:[0-9]+]] = OpConstant %[[uint]] 0
; CHECK-DAG: %[[uint_1:[0-9]+]] = OpConstant %[[uint]] 1
; CHECK-DAG: %[[coord_2darray:[0-9]+]] = OpConstantNull %[[v3uint]]
; CHECK-DAG: %[[offset_2d:[0-9]+]] = OpConstantComposite %[[v2uint]] %[[uint_1]] %[[uint_1]]

@.str = private unnamed_addr constant [6 x i8] c"img2d\00", align 1

define void @test_load_2darray() #0 {
entry:
  %img = tail call target("spirv.Image", float, 1, 2, 1, 0, 1, 0) @llvm.spv.resource.handlefrombinding.tspirv.Image_f32_1_2_1_0_1_0t(i32 0, i32 0, i32 1, i32 0, ptr @.str)
; CHECK: %[[img_val_2darray:[0-9]+]] = OpLoad %[[image_2darray]]
; CHECK: %[[res_2darray:[0-9]+]] = OpImageFetch %[[v4float]] %[[img_val_2darray]] %[[coord_2darray]] Lod %[[lod0]]
  %res = call <4 x float> @llvm.spv.resource.load.level.v4f32.tspirv.Image_f32_1_2_1_0_1_0t.v3i32.i32.v2i32(target("spirv.Image", float, 1, 2, 1, 0, 1, 0) %img, <3 x i32> zeroinitializer, i32 0, <2 x i32> zeroinitializer)
  ret void
}

define void @test_load_2darray_offset() #0 {
entry:
  %img = tail call target("spirv.Image", float, 1, 2, 1, 0, 1, 0) @llvm.spv.resource.handlefrombinding.tspirv.Image_f32_1_2_1_0_1_0t(i32 0, i32 0, i32 1, i32 0, ptr @.str)
; CHECK: %[[img_val_2darray_off:[0-9]+]] = OpLoad %[[image_2darray]]
; CHECK: %[[res_2darray_off:[0-9]+]] = OpImageFetch %[[v4float]] %[[img_val_2darray_off]] %[[coord_2darray]] Lod|ConstOffset %[[lod0]] %[[offset_2d]]
  %res = call <4 x float> @llvm.spv.resource.load.level.v4f32.tspirv.Image_f32_1_2_1_0_1_0t.v3i32.i32.v2i32(target("spirv.Image", float, 1, 2, 1, 0, 1, 0) %img, <3 x i32> zeroinitializer, i32 0, <2 x i32> <i32 1, i32 1>)
  ret void
}

attributes #0 = { "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" }

declare target("spirv.Image", float, 1, 2, 1, 0, 1, 0) @llvm.spv.resource.handlefrombinding.tspirv.Image_f32_1_2_1_0_1_0t(i32, i32, i32, i32, ptr)
declare <4 x float> @llvm.spv.resource.load.level.v4f32.tspirv.Image_f32_1_2_1_0_1_0t.v3i32.i32.v2i32(target("spirv.Image", float, 1, 2, 1, 0, 1, 0), <3 x i32>, i32, <2 x i32>)
