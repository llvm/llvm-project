; RUN: llc -mtriple=r600 -mcpu=cypress < %s | FileCheck %s

; CHECK-LABEL: {{^}}kernel_arg_i64:
define amdgpu_kernel void @kernel_arg_i64(ptr addrspace(1) %out, i64 %a) nounwind {
  store i64 %a, ptr addrspace(1) %out, align 8
  ret void
}

; i64 arg works, v1i64 arg does not.
; CHECK-LABEL: {{^}}kernel_arg_v1i64:
define amdgpu_kernel void @kernel_arg_v1i64(ptr addrspace(1) %out, <1 x i64> %a) nounwind {
  store <1 x i64> %a, ptr addrspace(1) %out, align 8
  ret void
}

