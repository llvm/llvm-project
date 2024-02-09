; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -verify-machineinstrs < %s | FileCheck %s

; Attribute not specified.
; CHECK-LABEL: {{^}}empty_no_attribute:
define amdgpu_kernel void @empty_no_attribute() {
entry:
  ret void
}

; Ignore if number of work groups for x dimension is 0.
; CHECK-LABEL: {{^}}empty_num_work_groups_x0:
define amdgpu_kernel void @empty_num_work_groups_x0() #0 {
entry:
  ret void
}
attributes #0 = {"amdgpu-num-work-groups"="0,2,3"}

; Ignore if number of work groups for y dimension is 0.
; CHECK-LABEL: {{^}}empty_num_work_groups_y0:
define amdgpu_kernel void @empty_num_work_groups_y0() #1 {
entry:
  ret void
}
attributes #1 = {"amdgpu-num-work-groups"="1,0,3"}

; Ignore if number of work groups for z dimension is 0.
; CHECK-LABEL: {{^}}empty_num_work_groups_z0:
define amdgpu_kernel void @empty_num_work_groups_z0() #2 {
entry:
  ret void
}
attributes #2 = {"amdgpu-num-work-groups"="1,2,0"}

; CHECK-LABEL: {{^}}empty_num_work_groups_1_2_3:
define amdgpu_kernel void @empty_num_work_groups_1_2_3() #3 {
entry:
  ret void
}
attributes #3 = {"amdgpu-num-work-groups"="1,2,3"}

; CHECK-LABEL: {{^}}empty_num_work_groups_1024_1024_1024:
define amdgpu_kernel void @empty_num_work_groups_1024_1024_1024() #4 {
entry:
  ret void
}
attributes #4 = {"amdgpu-num-work-groups"="1024,1024,1024"}


; CHECK: .amdgpu_metadata
; CHECK: - .args:
; CHECK:        .max_flat_workgroup_size: 1024
; CHECK-NEXT:   .name:           empty_no_attribute
; CHECK-NEXT:   .private_segment_fixed_size: 0

; CHECK: - .args:
; CHECK:        .max_flat_workgroup_size: 1024
; CHECK-NEXT:   .name:           empty_num_work_groups_x0
; CHECK-NEXT:   .private_segment_fixed_size: 0

; CHECK: - .args:
; CHECK:        .max_flat_workgroup_size: 1024
; CHECK-NEXT:   .name:           empty_num_work_groups_y0
; CHECK-NEXT:   .private_segment_fixed_size: 0

; CHECK: - .args:
; CHECK:        .max_flat_workgroup_size: 1024
; CHECK-NEXT:   .name:           empty_num_work_groups_z0
; CHECK-NEXT:   .private_segment_fixed_size: 0

; CHECK: - .args:
; CHECK:        .max_flat_workgroup_size: 1024
; CHECK-NEXT:   .name:           empty_num_work_groups_1_2_3
; CHECK-NEXT:   .num_work_groups_x: 1
; CHECK-NEXT:   .num_work_groups_y: 2
; CHECK-NEXT:   .num_work_groups_z: 3
; CHECK-NEXT:   .private_segment_fixed_size: 0

; CHECK: - .args:
; CHECK:        .max_flat_workgroup_size: 1024
; CHECK-NEXT:   .name:           empty_num_work_groups_1024_1024_1024
; CHECK-NEXT:   .num_work_groups_x: 1024
; CHECK-NEXT:   .num_work_groups_y: 1024
; CHECK-NEXT:   .num_work_groups_z: 1024
; CHECK-NEXT:   .private_segment_fixed_size: 0
