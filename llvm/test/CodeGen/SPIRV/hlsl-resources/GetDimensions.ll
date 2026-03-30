; RUN: llc -O0 -mtriple=spirv1.6-vulkan1.3-library %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv1.6-vulkan1.3-library %s -o - -filetype=obj | spirv-val --target-env vulkan1.3 %}

; CHECK-DAG: %[[int:[0-9]+]] = OpTypeInt 32 0
; CHECK-DAG: %[[v2int:[0-9]+]] = OpTypeVector %[[int]] 2
; CHECK-DAG: %[[float:[0-9]+]] = OpTypeFloat 32
; CHECK-DAG: %[[image:[0-9]+]] = OpTypeImage %[[float]] 2D 2 0 0 1 Unknown

@.str = private unnamed_addr constant [4 x i8] c"img\00", align 1

define void @main() #0 {
entry:
  %img = tail call target("spirv.Image", float, 1, 2, 0, 0, 1, 0) @llvm.spv.resource.handlefrombinding.tspirv.Image_f32_1_2_0_0_1_0t(i32 0, i32 0, i32 1, i32 0, ptr @.str)

; CHECK: %[[img_val:[0-9]+]] = OpLoad %[[image]]
; CHECK: %[[res0:[0-9]+]] = OpImageQuerySize %[[v2int]] %[[img_val]]
  %res0 = call <2 x i32> @llvm.spv.resource.get.dimensions.v2i32.tspirv.Image_f32_1_2_0_0_1_0t(target("spirv.Image", float, 1, 2, 0, 0, 1, 0) %img)

; CHECK: %[[img_val1:[0-9]+]] = OpLoad %[[image]]
; CHECK: %[[res1:[0-9]+]] = OpImageQuerySizeLod %[[v2int]] %[[img_val1]] %[[int]]
  %res1 = call <2 x i32> @llvm.spv.resource.get.dimensions.lod.v2i32.tspirv.Image_f32_1_2_0_0_1_0t(target("spirv.Image", float, 1, 2, 0, 0, 1, 0) %img, i32 0)

; CHECK: %[[img_val2:[0-9]+]] = OpLoad %[[image]]
; CHECK: %[[res2:[0-9]+]] = OpImageQueryLevels %[[int]] %[[img_val2]]
  %res2 = call i32 @llvm.spv.resource.get.levels.i32.tspirv.Image_f32_1_2_0_0_1_0t(target("spirv.Image", float, 1, 2, 0, 0, 1, 0) %img)

  ret void
}

attributes #0 = { "hlsl.shader"="pixel" }

declare target("spirv.Image", float, 1, 2, 0, 0, 1, 0) @llvm.spv.resource.handlefrombinding.tspirv.Image_f32_1_2_0_0_1_0t(i32, i32, i32, i32, ptr)
declare <2 x i32> @llvm.spv.resource.get.dimensions.v2i32.tspirv.Image_f32_1_2_0_0_1_0t(target("spirv.Image", float, 1, 2, 0, 0, 1, 0))
declare <2 x i32> @llvm.spv.resource.get.dimensions.lod.v2i32.tspirv.Image_f32_1_2_0_0_1_0t(target("spirv.Image", float, 1, 2, 0, 0, 1, 0), i32)
declare i32 @llvm.spv.resource.get.levels.i32.tspirv.Image_f32_1_2_0_0_1_0t(target("spirv.Image", float, 1, 2, 0, 0, 1, 0))
