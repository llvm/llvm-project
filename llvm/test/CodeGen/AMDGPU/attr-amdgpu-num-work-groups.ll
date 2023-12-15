; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -verify-machineinstrs < %s | FileCheck %s

; Attribute not specified.
; CHECK-LABEL: {{^}}empty_no_attribute:
define amdgpu_kernel void @empty_no_attribute() {
entry:
  ret void
}

; Ignore if number of work groups is 0.
; CHECK-LABEL: {{^}}empty_num_work_groups_0:
define amdgpu_kernel void @empty_num_work_groups_0() #0 {
entry:
  ret void
}
attributes #0 = {"amdgpu-num-work-groups"="0"}

; Exactly 1 work group.
; CHECK-LABEL: {{^}}empty_num_work_groups_1:
define amdgpu_kernel void @empty_num_work_groups_1() #1 {
entry:
  ret void
}
attributes #1 = {"amdgpu-num-work-groups"="1"}

; Exactly 5 work groups.
; CHECK-LABEL: {{^}}empty_num_work_groups_5:
define amdgpu_kernel void @empty_num_work_groups_5() #2 {
entry:
  ret void
}
attributes #2 = {"amdgpu-num-work-groups"="5"}

; Exactly 32 work groups.
; CHECK-LABEL: {{^}}empty_num_work_groups_32:
define amdgpu_kernel void @empty_num_work_groups_32() #3 {
entry:
  ret void
}
attributes #3 = {"amdgpu-num-work-groups"="32"}

; Exactly 50 work groups.
; CHECK-LABEL: {{^}}empty_num_work_groups_50:
define amdgpu_kernel void @empty_num_work_groups_50() #4 {
entry:
  ret void
}
attributes #4 = {"amdgpu-num-work-groups"="50"}

; Exactly 256 work groups.
; CHECK-LABEL: {{^}}empty_num_work_groups_256:
define amdgpu_kernel void @empty_num_work_groups_256() #5 {
entry:
  ret void
}
attributes #5 = {"amdgpu-num-work-groups"="256"}

; Exactly 1024 work groups.
; CHECK-LABEL: {{^}}empty_num_work_groups_1024:
define amdgpu_kernel void @empty_num_work_groups_1024() #6 {
entry:
  ret void
}
attributes #6 = {"amdgpu-num-work-groups"="1024"}

; CHECK: .amdgpu_metadata
; CHECK:        .name:           empty_no_attribute
; CHECK-NEXT:   .private_segment_fixed_size: 0
; CHECK:        .name:           empty_num_work_groups_0
; CHECK-NEXT:   .private_segment_fixed_size: 0
; CHECK:        .name:           empty_num_work_groups_1
; CHECK-NEXT:   .num_work_groups: 1
; CHECK:        .name:           empty_num_work_groups_5
; CHECK-NEXT:   .num_work_groups: 5
; CHECK:        .name:           empty_num_work_groups_32
; CHECK-NEXT:   .num_work_groups: 32
; CHECK:        .name:           empty_num_work_groups_50
; CHECK-NEXT:   .num_work_groups: 50
; CHECK:        .name:           empty_num_work_groups_256
; CHECK-NEXT:   .num_work_groups: 256
; CHECK:        .name:           empty_num_work_groups_1024
; CHECK-NEXT:   .num_work_groups: 1024
