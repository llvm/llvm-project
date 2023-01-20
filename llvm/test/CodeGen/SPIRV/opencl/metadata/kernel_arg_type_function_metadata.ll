; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s

; CHECK: %[[#TypeSampler:]] = OpTypeSampler
define spir_kernel void @foo(i64 %sampler) !kernel_arg_addr_space !7 !kernel_arg_access_qual !8 !kernel_arg_type !9 !kernel_arg_type_qual !10 !kernel_arg_base_type !9 {
entry:
  ret void
}

!7 = !{i32 0}
!8 = !{!"none"}
!9 = !{!"sampler_t"}
!10 = !{!""}
