; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV
; TODO: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-SPIRV: %[[#IMAGE_TYPE:]] = OpTypeImage
; CHECK-SPIRV: %[[#IMAGE_ARG:]] = OpFunctionParameter %[[#IMAGE_TYPE]]
; CHECK-SPIRV: %[[#]] = OpImageQuerySizeLod %[[#]] %[[#IMAGE_ARG]]
; CHECK-SPIRV: %[[#]] = OpImageQuerySizeLod %[[#]] %[[#IMAGE_ARG]]

define spir_kernel void @sample_kernel(target("spirv.Image", void, 1, 0, 1, 0, 0, 0, 0) %input) {
entry:
  %call.tmp1 = call spir_func <2 x i32> @_Z13get_image_dim20ocl_image2d_array_ro(target("spirv.Image", void, 1, 0, 1, 0, 0, 0, 0) %input)
  %call.tmp2 = shufflevector <2 x i32> %call.tmp1, <2 x i32> undef, <3 x i32> <i32 0, i32 1, i32 2>
  %call.tmp3 = call spir_func i64 @_Z20get_image_array_size20ocl_image2d_array_ro(target("spirv.Image", void, 1, 0, 1, 0, 0, 0, 0) %input)
  %call.tmp4 = trunc i64 %call.tmp3 to i32
  %call.tmp5 = insertelement <3 x i32> %call.tmp2, i32 %call.tmp4, i32 2
  %call.old = extractelement <3 x i32> %call.tmp5, i32 0
  ret void
}

declare spir_func <2 x i32> @_Z13get_image_dim20ocl_image2d_array_ro(target("spirv.Image", void, 1, 0, 1, 0, 0, 0, 0))

declare spir_func i64 @_Z20get_image_array_size20ocl_image2d_array_ro(target("spirv.Image", void, 1, 0, 1, 0, 0, 0, 0))
