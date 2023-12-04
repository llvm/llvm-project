; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx90a < %s | FileCheck %s

; Make sure that SYCL kernels with less than 3 dimensions specified in required
; work group size, have those dimensions padded up with 1.

; CHECK-LABEL: .name:           sycl_kernel_1dim
; CHECK:    .reqd_workgroup_size:
; CHECK-NEXT:      - 3
; CHECK-NEXT:      - 1
; CHECK-NEXT:      - 1
define protected amdgpu_kernel void @sycl_kernel_1dim() #1 !reqd_work_group_size !0 {
entry:
  ret void
}

; CHECK-LABEL: .name:           sycl_kernel_2dim
; CHECK:    .reqd_workgroup_size:
; CHECK-NEXT:      - 5
; CHECK-NEXT:      - 7
; CHECK-NEXT:      - 1
define protected amdgpu_kernel void @sycl_kernel_2dim() #1 !reqd_work_group_size !1 {
entry:
  ret void
}

; CHECK-LABEL: .name:           sycl_kernel_3dim
; CHECK:    .reqd_workgroup_size:
; CHECK-NEXT:      - 11 
; CHECK-NEXT:      - 13
; CHECK-NEXT:      - 17
define protected amdgpu_kernel void @sycl_kernel_3dim() #1 !reqd_work_group_size !2 {
entry:
  ret void
}

attributes #0 = { nounwind speculatable memory(none) }
attributes #1 = { "sycl-module-id"="reqd_work_group_size_check_exception.cpp" }


!0 = !{i32 3}
!1 = !{i32 5, i32 7}
!2 = !{i32 11, i32 13, i32 17}
