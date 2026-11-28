; RUN: llc < %s -mtriple=amdgpu6.00 | FileCheck --check-prefix=SI %s

; Make sure the OpenCL Image lowering pass doesn't crash when argument metadata
; is not in expected order.

; SI: s_endpgm
define amdgpu_kernel void @kernel(ptr addrspace(1) %out) {
entry:
  store i32 0, ptr addrspace(1) %out
  ret void
}

attributes #3 = { nounwind }

!opencl.kernels = !{!0}

!0 = !{ptr @kernel, !1, !2, !3, !4, !5}
!1 = !{!"kernel_arg_addr_space", i32 0}
!2 = !{!"kernel_arg_access_qual", !"none"}
!3 = !{!"kernel_arg_type", !"int*"}
!4 = !{!"kernel_arg_type_qual", !""}
!5 = !{!"kernel_arg_name", !""}
