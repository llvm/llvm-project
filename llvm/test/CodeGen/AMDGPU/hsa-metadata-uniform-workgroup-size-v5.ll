; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -filetype=obj -o - < %s | llvm-readelf --notes - | FileCheck %s
; RUN: llc -mtriple=amdgcn--amdhsa -mcpu=gfx900 < %s | FileCheck %s

; CHECK: ---
; CHECK: amdhsa.kernels:
; CHECK:  - .args:
; CHECK-LABEL:     .name:           kernel_uniform_workgroup
; CHECK:     .uniform_work_group_size: 1
define amdgpu_kernel void @kernel_uniform_workgroup() "uniform-work-group-size"="true" {
bb:
  ret void
}

; CHECK:  - .args:
; CHECK-LABEL:     .name:           kernel_non_uniform_workgroup
; CHECK-NOT:     .uniform_work_group_size:
define amdgpu_kernel void @kernel_non_uniform_workgroup() "uniform-work-group-size"="false" {
bb:
  ret void
}

; CHECK:  - .args:
; CHECK-LABEL:     .name:           kernel_no_attr
; CHECK-NOT:     .uniform_work_group_size:
define amdgpu_kernel void @kernel_no_attr() {
bb:
  ret void
}

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"amdhsa_code_object_version", i32 500}
