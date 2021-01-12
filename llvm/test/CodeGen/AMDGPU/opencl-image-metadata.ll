; RUN: llc < %s -march=amdgcn -verify-machineinstrs | FileCheck --check-prefix=SI %s
; RUN: llc < %s -march=r600 -mcpu=redwood | FileCheck --check-prefix=EG %s

; Make sure the OpenCL Image lowering pass doesn't crash when argument metadata
; is not in expected order.

; EG: CF_END
; SI: s_endpgm
define amdgpu_kernel void @kernel(i32 addrspace(1)* %out) {
entry:
  store i32 0, i32 addrspace(1)* %out
  ret void
}

attributes #3 = { nounwind }

!opencl.kernels = !{!0}

!0 = !{void (i32 addrspace(1)*)* @kernel, !1, !2, !3, !4, !5}
!1 = !{!"kernel_arg_addr_space", i32 0}
!2 = !{!"kernel_arg_access_qual", !"none"}
!3 = !{!"kernel_arg_type", !"int*"}
!4 = !{!"kernel_arg_type_qual", !""}
!5 = !{!"kernel_arg_name", !""}
