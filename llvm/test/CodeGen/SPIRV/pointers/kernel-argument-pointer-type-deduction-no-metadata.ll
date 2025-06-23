; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

define spir_kernel void @test(ptr addrspace(1) %srcimg) {
; CHECK: %[[#VOID:]] = OpTypeVoid
; CHECK: %[[#IMAGE:]] = OpTypeImage %[[#VOID]] 2D 0 0 0 0 Unknown ReadOnly
; CHECK: %[[#PARAM:]] = OpFunctionParameter %[[#IMAGE]]
; CHECK: %[[#]] = OpImageQuerySizeLod %[[#]] %[[#PARAM]] %[[#]]
  %call = call spir_func <2 x i32> @_Z13get_image_dim14ocl_image2d_ro(ptr addrspace(1) %srcimg)
  ret void
}

declare spir_func <2 x i32> @_Z13get_image_dim14ocl_image2d_ro(ptr addrspace(1))
