; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s

; CHECK: %[[#TypeSampler:]] = OpTypeSampler
define spir_kernel void @foo(i64 %sampler) {
entry:
  ret void
}
!opencl.kernels = !{!0}

!0 = !{void (i64)* @foo, !1, !2, !3, !4, !5, !6}
!1 = !{!"kernel_arg_addr_space", i32 0}
!2 = !{!"kernel_arg_access_qual", !"none"}
!3 = !{!"kernel_arg_type", !"sampler_t"}
!4 = !{!"kernel_arg_type_qual", !""}
!5 = !{!"kernel_arg_base_type", !"sampler_t"}
!6 = !{!"kernel_arg_name", !"sampler"}
