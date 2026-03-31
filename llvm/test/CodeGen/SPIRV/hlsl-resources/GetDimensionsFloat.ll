; RUN: llc -O0 -mtriple=spirv1.6-vulkan1.3-library %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv1.6-vulkan1.3-library %s -o - -filetype=obj | spirv-val --target-env vulkan1.3 %}

; CHECK-DAG: [[FLOAT_TY:%[0-9]+]] = OpTypeFloat 32
; CHECK-DAG: [[IMG_TY:%[0-9]+]] = OpTypeImage [[FLOAT_TY]] 1D 2 0 0 2 Unknown
; CHECK-DAG: [[INT_TY:%[0-9]+]] = OpTypeInt 32 0
; CHECK-DAG: [[PTR_IMG_TY:%[0-9]+]] = OpTypePointer UniformConstant [[IMG_TY]]
; CHECK: [[VAR:%[0-9]+]] = OpVariable [[PTR_IMG_TY]] UniformConstant
; CHECK: [[LOAD:%[0-9]+]] = OpLoad [[IMG_TY]] [[VAR]]
; CHECK: [[QUERY:%[0-9]+]] = OpImageQuerySize [[INT_TY]] [[LOAD]]
; CHECK: {{%[0-9]+}} = OpConvertUToF [[FLOAT_TY]] [[QUERY]]

@.str1 = private unnamed_addr constant [6 x i8] c"img1d\00", align 1

define void @main() #0 {
entry:
  %img1d = tail call target("spirv.Image", float, 0, 2, 0, 0, 2, 0) @llvm.spv.resource.handlefrombinding.tspirv.Image_f32_0_2_0_0_2_0t(i32 0, i32 0, i32 1, i32 0, ptr @.str1)
  %res0 = call float @llvm.spv.resource.getdimensions.x.f32.tspirv.Image_f32_0_2_0_0_2_0t(target("spirv.Image", float, 0, 2, 0, 0, 2, 0) %img1d)
  ret void
}

attributes #0 = { "hlsl.shader"="pixel" }

declare target("spirv.Image", float, 0, 2, 0, 0, 2, 0) @llvm.spv.resource.handlefrombinding.tspirv.Image_f32_0_2_0_0_2_0t(i32, i32, i32, i32, ptr)
declare float @llvm.spv.resource.getdimensions.x.f32.tspirv.Image_f32_0_2_0_0_2_0t(target("spirv.Image", float, 0, 2, 0, 0, 2, 0))
