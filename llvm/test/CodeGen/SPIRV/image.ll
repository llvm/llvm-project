; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV

; CHECK-SPIRV:     %[[#VOID_TY:]] = OpTypeVoid
; CHECK-SPIRV-DAG: %[[#]] = OpTypeImage %[[#VOID_TY]] 2D 0 0 0 0 Unknown ReadOnly
; CHECK-SPIRV-DAG: %[[#]] = OpTypeImage %[[#VOID_TY]] 2D 0 0 0 0 Unknown WriteOnly
; CHECK-SPIRV-NOT: %[[#]] = OpTypeImage %[[#VOID_TY]] 2D 0 0 0 0 Unknown ReadOnly
; CHECK-SPIRV:     OpImageSampleExplicitLod
; CHECK-SPIRV:     OpImageWrite

define spir_kernel void @image_copy(target("spirv.Image", void, 1, 0, 0, 0, 0, 0, 0) %image1, target("spirv.Image", void, 1, 0, 0, 0, 0, 0, 0) %image2) !kernel_arg_access_qual !1 {
  %call = tail call spir_func i64 @_Z13get_global_idj(i32 0)
  %conv = trunc i64 %call to i32
  %call1 = tail call spir_func i64 @_Z13get_global_idj(i32 1)
  %conv2 = trunc i64 %call1 to i32
  %vecinit = insertelement <2 x i32> undef, i32 %conv, i32 0
  %vecinit3 = insertelement <2 x i32> %vecinit, i32 %conv2, i32 1
  %call4 = tail call spir_func <4 x float> @_Z11read_imagef11ocl_image2d11ocl_samplerDv2_i(target("spirv.Image", void, 1, 0, 0, 0, 0, 0, 0) %image1, i32 20, <2 x i32> %vecinit3)
  tail call spir_func void @_Z12write_imagef11ocl_image2dDv2_iDv4_f(target("spirv.Image", void, 1, 0, 0, 0, 0, 0, 0) %image2, <2 x i32> %vecinit3, <4 x float> %call4)
  ret void
}

declare spir_func i64 @_Z13get_global_idj(i32)

declare spir_func <4 x float> @_Z11read_imagef11ocl_image2d11ocl_samplerDv2_i(target("spirv.Image", void, 1, 0, 0, 0, 0, 0, 0), i32, <2 x i32>)

declare spir_func void @_Z12write_imagef11ocl_image2dDv2_iDv4_f(target("spirv.Image", void, 1, 0, 0, 0, 0, 0, 0), <2 x i32>, <4 x float>)

!1 = !{!"read_only", !"write_only"}
