; RUN: llc -O0 -mtriple=spirv1.6-vulkan1.3-library %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv1.6-vulkan1.3-library %s -o - -filetype=obj | spirv-val --target-env vulkan1.3 %}

; CHECK-DAG: %[[int:[0-9]+]] = OpTypeInt 32 0
; CHECK-DAG: %[[v2int:[0-9]+]] = OpTypeVector %[[int]] 2
; CHECK-DAG: %[[v3int:[0-9]+]] = OpTypeVector %[[int]] 3
; CHECK-DAG: %[[v4int:[0-9]+]] = OpTypeVector %[[int]] 4
; CHECK-DAG: %[[float:[0-9]+]] = OpTypeFloat 32
; CHECK-DAG: %[[image1d_s1:[0-9]+]] = OpTypeImage %[[float]] 1D 2 0 0 1 Unknown
; CHECK-DAG: %[[image2d_s1:[0-9]+]] = OpTypeImage %[[float]] 2D 2 0 0 1 Unknown
; CHECK-DAG: %[[image3d_s1:[0-9]+]] = OpTypeImage %[[float]] 3D 2 0 0 1 Unknown
; CHECK-DAG: %[[image1d_s2:[0-9]+]] = OpTypeImage %[[float]] 1D 2 0 0 2 Unknown
; CHECK-DAG: %[[image2d_s2:[0-9]+]] = OpTypeImage %[[float]] 2D 2 0 0 2 Unknown
; CHECK-DAG: %[[image3d_s2:[0-9]+]] = OpTypeImage %[[float]] 3D 2 0 0 2 Unknown
; CHECK-DAG: %[[imagems:[0-9]+]] = OpTypeImage %[[float]] 2D 2 0 1 1 Unknown
; CHECK-DAG: %[[imagemsarray:[0-9]+]] = OpTypeImage %[[float]] 2D 2 1 1 1 Unknown
; CHECK-DAG: %[[int_0:[0-9]+]] = OpConstant %[[int]] 0

@.str1 = private unnamed_addr constant [6 x i8] c"img1d\00", align 1
@.str2 = private unnamed_addr constant [6 x i8] c"img2d\00", align 1
@.str3 = private unnamed_addr constant [6 x i8] c"img3d\00", align 1
@.str4 = private unnamed_addr constant [9 x i8] c"img1dlod\00", align 1
@.str5 = private unnamed_addr constant [9 x i8] c"img2dlod\00", align 1
@.str6 = private unnamed_addr constant [9 x i8] c"img3dlod\00", align 1
@.str7 = private unnamed_addr constant [6 x i8] c"imgms\00", align 1
@.str8 = private unnamed_addr constant [11 x i8] c"imgmsarray\00", align 1

define void @main() #0 {
entry:
  %img1d = tail call target("spirv.Image", float, 0, 2, 0, 0, 2, 0) @llvm.spv.resource.handlefrombinding.tspirv.Image_f32_0_2_0_0_2_0t(i32 0, i32 0, i32 1, i32 0, ptr @.str1)
  %img2d = tail call target("spirv.Image", float, 1, 2, 0, 0, 2, 0) @llvm.spv.resource.handlefrombinding.tspirv.Image_f32_1_2_0_0_2_0t(i32 0, i32 1, i32 1, i32 0, ptr @.str2)
  %img3d = tail call target("spirv.Image", float, 2, 2, 0, 0, 2, 0) @llvm.spv.resource.handlefrombinding.tspirv.Image_f32_2_2_0_0_2_0t(i32 0, i32 2, i32 1, i32 0, ptr @.str3)
  
  %img1dlod = tail call target("spirv.Image", float, 0, 2, 0, 0, 1, 0) @llvm.spv.resource.handlefrombinding.tspirv.Image_f32_0_2_0_0_1_0t(i32 0, i32 3, i32 1, i32 0, ptr @.str4)
  %img2dlod = tail call target("spirv.Image", float, 1, 2, 0, 0, 1, 0) @llvm.spv.resource.handlefrombinding.tspirv.Image_f32_1_2_0_0_1_0t(i32 0, i32 4, i32 1, i32 0, ptr @.str5)
  %img3dlod = tail call target("spirv.Image", float, 2, 2, 0, 0, 1, 0) @llvm.spv.resource.handlefrombinding.tspirv.Image_f32_2_2_0_0_1_0t(i32 0, i32 5, i32 1, i32 0, ptr @.str6)

  %imgms = tail call target("spirv.Image", float, 1, 2, 0, 1, 1, 0) @llvm.spv.resource.handlefrombinding.tspirv.Image_f32_1_2_0_1_1_0t(i32 0, i32 6, i32 1, i32 0, ptr @.str7)
  %imgmsarray = tail call target("spirv.Image", float, 1, 2, 1, 1, 1, 0) @llvm.spv.resource.handlefrombinding.tspirv.Image_f32_1_2_1_1_1_0t(i32 0, i32 7, i32 1, i32 0, ptr @.str8)

; CHECK: %[[img1d_val:[0-9]+]] = OpLoad %[[image1d_s2]]
; CHECK: OpImageQuerySize %[[int]] %[[img1d_val]]
  %res0 = call i32 @llvm.spv.resource.getdimensions.x.tspirv.Image_f32_0_2_0_0_2_0t(target("spirv.Image", float, 0, 2, 0, 0, 2, 0) %img1d)

; CHECK: %[[img2d_val:[0-9]+]] = OpLoad %[[image2d_s2]]
; CHECK: OpImageQuerySize %[[v2int]] %[[img2d_val]]
  %res1 = call <2 x i32> @llvm.spv.resource.getdimensions.xy.tspirv.Image_f32_1_2_0_0_2_0t(target("spirv.Image", float, 1, 2, 0, 0, 2, 0) %img2d)

; CHECK: %[[img3d_val:[0-9]+]] = OpLoad %[[image3d_s2]]
; CHECK: OpImageQuerySize %[[v3int]] %[[img3d_val]]
  %res2 = call <3 x i32> @llvm.spv.resource.getdimensions.xyz.tspirv.Image_f32_2_2_0_0_2_0t(target("spirv.Image", float, 2, 2, 0, 0, 2, 0) %img3d)

; CHECK: %[[img1dlod_val:[0-9]+]] = OpLoad %[[image1d_s1]]
; CHECK: %[[size3:[0-9]+]] = OpImageQuerySizeLod %[[int]] %[[img1dlod_val]] %[[int_0]]
; CHECK: %[[levels3:[0-9]+]] = OpImageQueryLevels %[[int]] %[[img1dlod_val]]
; CHECK: OpCompositeConstruct %[[v2int]] %[[size3]] %[[levels3]]
  %res3 = call <2 x i32> @llvm.spv.resource.getdimensions.levels.x.tspirv.Image_f32_0_2_0_0_1_0t(target("spirv.Image", float, 0, 2, 0, 0, 1, 0) %img1dlod, i32 0)

; CHECK: %[[img2dlod_val:[0-9]+]] = OpLoad %[[image2d_s1]]
; CHECK: %[[size4:[0-9]+]] = OpImageQuerySizeLod %[[v2int]] %[[img2dlod_val]] %[[int_0]]
; CHECK: %[[levels4:[0-9]+]] = OpImageQueryLevels %[[int]] %[[img2dlod_val]]
; CHECK: OpCompositeConstruct %[[v3int]] %[[size4]] %[[levels4]]
  %res4 = call <3 x i32> @llvm.spv.resource.getdimensions.levels.xy.tspirv.Image_f32_1_2_0_0_1_0t(target("spirv.Image", float, 1, 2, 0, 0, 1, 0) %img2dlod, i32 0)

; CHECK: %[[img3dlod_val:[0-9]+]] = OpLoad %[[image3d_s1]]
; CHECK: %[[size5:[0-9]+]] = OpImageQuerySizeLod %[[v3int]] %[[img3dlod_val]] %[[int_0]]
; CHECK: %[[levels5:[0-9]+]] = OpImageQueryLevels %[[int]] %[[img3dlod_val]]
; CHECK: OpCompositeConstruct %[[v4int]] %[[size5]] %[[levels5]]
  %res5 = call <4 x i32> @llvm.spv.resource.getdimensions.levels.xyz.tspirv.Image_f32_2_2_0_0_1_0t(target("spirv.Image", float, 2, 2, 0, 0, 1, 0) %img3dlod, i32 0)

; CHECK: %[[imgms_val:[0-9]+]] = OpLoad %[[imagems]]
; CHECK: %[[size6:[0-9]+]] = OpImageQuerySize %[[v2int]] %[[imgms_val]]
; CHECK: %[[samp6:[0-9]+]] = OpImageQuerySamples %[[int]] %[[imgms_val]]
; CHECK: OpCompositeConstruct %[[v3int]] %[[size6]] %[[samp6]]
  %res6 = call <3 x i32> @llvm.spv.resource.getdimensions.ms.xy.tspirv.Image_f32_1_2_0_1_1_0t(target("spirv.Image", float, 1, 2, 0, 1, 1, 0) %imgms)

; CHECK: %[[imgmsarray_val:[0-9]+]] = OpLoad %[[imagemsarray]]
; CHECK: %[[size7:[0-9]+]] = OpImageQuerySize %[[v3int]] %[[imgmsarray_val]]
; CHECK: %[[samp7:[0-9]+]] = OpImageQuerySamples %[[int]] %[[imgmsarray_val]]
; CHECK: OpCompositeConstruct %[[v4int]] %[[size7]] %[[samp7]]
  %res7 = call <4 x i32> @llvm.spv.resource.getdimensions.ms.xyz.tspirv.Image_f32_1_2_1_1_1_0t(target("spirv.Image", float, 1, 2, 1, 1, 1, 0) %imgmsarray)

  ret void
}

attributes #0 = { "hlsl.shader"="pixel" }

declare target("spirv.Image", float, 0, 2, 0, 0, 2, 0) @llvm.spv.resource.handlefrombinding.tspirv.Image_f32_0_2_0_0_2_0t(i32, i32, i32, i32, ptr)
declare target("spirv.Image", float, 1, 2, 0, 0, 2, 0) @llvm.spv.resource.handlefrombinding.tspirv.Image_f32_1_2_0_0_2_0t(i32, i32, i32, i32, ptr)
declare target("spirv.Image", float, 2, 2, 0, 0, 2, 0) @llvm.spv.resource.handlefrombinding.tspirv.Image_f32_2_2_0_0_2_0t(i32, i32, i32, i32, ptr)
declare target("spirv.Image", float, 0, 2, 0, 0, 1, 0) @llvm.spv.resource.handlefrombinding.tspirv.Image_f32_0_2_0_0_1_0t(i32, i32, i32, i32, ptr)
declare target("spirv.Image", float, 1, 2, 0, 0, 1, 0) @llvm.spv.resource.handlefrombinding.tspirv.Image_f32_1_2_0_0_1_0t(i32, i32, i32, i32, ptr)
declare target("spirv.Image", float, 2, 2, 0, 0, 1, 0) @llvm.spv.resource.handlefrombinding.tspirv.Image_f32_2_2_0_0_1_0t(i32, i32, i32, i32, ptr)
declare target("spirv.Image", float, 1, 2, 0, 1, 1, 0) @llvm.spv.resource.handlefrombinding.tspirv.Image_f32_1_2_0_1_1_0t(i32, i32, i32, i32, ptr)
declare target("spirv.Image", float, 1, 2, 1, 1, 1, 0) @llvm.spv.resource.handlefrombinding.tspirv.Image_f32_1_2_1_1_1_0t(i32, i32, i32, i32, ptr)

declare i32 @llvm.spv.resource.getdimensions.x.tspirv.Image_f32_0_2_0_0_2_0t(target("spirv.Image", float, 0, 2, 0, 0, 2, 0))
declare <2 x i32> @llvm.spv.resource.getdimensions.xy.tspirv.Image_f32_1_2_0_0_2_0t(target("spirv.Image", float, 1, 2, 0, 0, 2, 0))
declare <3 x i32> @llvm.spv.resource.getdimensions.xyz.tspirv.Image_f32_2_2_0_0_2_0t(target("spirv.Image", float, 2, 2, 0, 0, 2, 0))

declare <2 x i32> @llvm.spv.resource.getdimensions.levels.x.tspirv.Image_f32_0_2_0_0_1_0t(target("spirv.Image", float, 0, 2, 0, 0, 1, 0), i32)
declare <3 x i32> @llvm.spv.resource.getdimensions.levels.xy.tspirv.Image_f32_1_2_0_0_1_0t(target("spirv.Image", float, 1, 2, 0, 0, 1, 0), i32)
declare <4 x i32> @llvm.spv.resource.getdimensions.levels.xyz.tspirv.Image_f32_2_2_0_0_1_0t(target("spirv.Image", float, 2, 2, 0, 0, 1, 0), i32)

declare <3 x i32> @llvm.spv.resource.getdimensions.ms.xy.tspirv.Image_f32_1_2_0_1_1_0t(target("spirv.Image", float, 1, 2, 0, 1, 1, 0))
declare <4 x i32> @llvm.spv.resource.getdimensions.ms.xyz.tspirv.Image_f32_1_2_1_1_1_0t(target("spirv.Image", float, 1, 2, 1, 1, 1, 0))
