; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s

; CHECK-DAG: %[[#IMAGE:]] = OpTypeImage %2 2D 0 0 0 0 Unknown ReadOnly

; CHECK: %[[#PARAM:]] = OpFunctionParameter %[[#IMAGE:]]
; CHECK-NOT: OpBitcast
; CHECK: %[[#]] = OpImageQuerySizeLod %[[#]] %[[#PARAM]] %[[#]]

define spir_kernel void @test(ptr addrspace(1) %srcimg) #0 !kernel_arg_addr_space !1 !kernel_arg_access_qual !2 !kernel_arg_type !3 !kernel_arg_type_qual !4 !kernel_arg_base_type !3 {
  %call = call spir_func <2 x i32> @_Z13get_image_dim14ocl_image2d_ro(ptr addrspace(1) %srcimg)
  ret void
}

declare spir_func <2 x i32> @_Z13get_image_dim14ocl_image2d_ro(ptr addrspace(1))

!1 = !{i32 1}
!2 = !{!"read_only"}
!3 = !{!"image2d_t"}
!4 = !{!""}
