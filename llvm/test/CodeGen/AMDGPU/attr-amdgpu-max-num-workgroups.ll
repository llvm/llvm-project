; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 < %s | FileCheck %s

; Attribute not specified.
; CHECK-LABEL: {{^}}empty_no_attribute:
define amdgpu_kernel void @empty_no_attribute() {
entry:
  ret void
}

; Ignore if number of work groups for x dimension is 0.
; CHECK-LABEL: {{^}}empty_max_num_workgroups_x0:
define amdgpu_kernel void @empty_max_num_workgroups_x0() #0 {
entry:
  ret void
}
attributes #0 = {"amdgpu-max-num-workgroups"="0,2,3"}

; Ignore if number of work groups for y dimension is 0.
; CHECK-LABEL: {{^}}empty_max_num_workgroups_y0:
define amdgpu_kernel void @empty_max_num_workgroups_y0() #1 {
entry:
  ret void
}
attributes #1 = {"amdgpu-max-num-workgroups"="1,0,3"}

; Ignore if number of work groups for z dimension is 0.
; CHECK-LABEL: {{^}}empty_max_num_workgroups_z0:
define amdgpu_kernel void @empty_max_num_workgroups_z0() #2 {
entry:
  ret void
}
attributes #2 = {"amdgpu-max-num-workgroups"="1,2,0"}

; CHECK-LABEL: {{^}}empty_max_num_workgroups_1_2_3:
define amdgpu_kernel void @empty_max_num_workgroups_1_2_3() #3 {
entry:
  ret void
}
attributes #3 = {"amdgpu-max-num-workgroups"="1,2,3"}

; CHECK-LABEL: {{^}}empty_max_num_workgroups_1024_1024_1024:
define amdgpu_kernel void @empty_max_num_workgroups_1024_1024_1024() #4 {
entry:
  ret void
}
attributes #4 = {"amdgpu-max-num-workgroups"="1024,1024,1024"}



; Ignore if number of work groups for x dimension is 0.
; CHECK-LABEL: {{^}}empty_max_num_workgroups_x_max:
define amdgpu_kernel void @empty_max_num_workgroups_x_max() #5 {
entry:
  ret void
}
attributes #5 = {"amdgpu-max-num-workgroups"="4294967295,2,3"}

; Ignore if number of work groups for y dimension is 0.
; CHECK-LABEL: {{^}}empty_max_num_workgroups_y_max:
define amdgpu_kernel void @empty_max_num_workgroups_y_max() #6 {
entry:
  ret void
}
attributes #6 = {"amdgpu-max-num-workgroups"="1,4294967295,3"}

; Ignore if number of work groups for z dimension is 0.
; CHECK-LABEL: {{^}}empty_max_num_workgroups_z_max:
define amdgpu_kernel void @empty_max_num_workgroups_z_max() #7 {
entry:
  ret void
}
attributes #7 = {"amdgpu-max-num-workgroups"="1,2,4294967295"}


; CHECK: .amdgpu_metadata
; CHECK: - .args:
; CHECK:        .max_flat_workgroup_size: 1024
; CHECK-NEXT:   .name:           empty_no_attribute
; CHECK-NEXT:   .private_segment_fixed_size: 0

; CHECK: - .args:
; CHECK:        .max_flat_workgroup_size: 1024
; CHECK-NEXT:   .max_num_workgroups_y: 2
; CHECK-NEXT:   .max_num_workgroups_z: 3
; CHECK-NEXT:   .name:           empty_max_num_workgroups_x0
; CHECK-NEXT:   .private_segment_fixed_size: 0

; CHECK: - .args:
; CHECK:        .max_flat_workgroup_size: 1024
; CHECK-NEXT:   .max_num_workgroups_x: 1
; CHECK-NEXT:   .max_num_workgroups_z: 3
; CHECK-NEXT:   .name:           empty_max_num_workgroups_y0
; CHECK-NEXT:   .private_segment_fixed_size: 0

; CHECK: - .args:
; CHECK:        .max_flat_workgroup_size: 1024
; CHECK-NEXT:   .max_num_workgroups_x: 1
; CHECK-NEXT:   .max_num_workgroups_y: 2
; CHECK-NEXT:   .name:           empty_max_num_workgroups_z0
; CHECK-NEXT:   .private_segment_fixed_size: 0

; CHECK: - .args:
; CHECK:        .max_flat_workgroup_size: 1024
; CHECK-NEXT:   .max_num_workgroups_x: 1
; CHECK-NEXT:   .max_num_workgroups_y: 2
; CHECK-NEXT:   .max_num_workgroups_z: 3
; CHECK-NEXT:   .name:           empty_max_num_workgroups_1_2_3
; CHECK-NEXT:   .private_segment_fixed_size: 0

; CHECK: - .args:
; CHECK:        .max_flat_workgroup_size: 1024
; CHECK-NEXT:   .max_num_workgroups_x: 1024
; CHECK-NEXT:   .max_num_workgroups_y: 1024
; CHECK-NEXT:   .max_num_workgroups_z: 1024
; CHECK-NEXT:   .name:           empty_max_num_workgroups_1024_1024_1024
; CHECK-NEXT:   .private_segment_fixed_size: 0


; CHECK: - .args:
; CHECK:        .max_flat_workgroup_size: 1024
; CHECK-NEXT:   .max_num_workgroups_y: 2
; CHECK-NEXT:   .max_num_workgroups_z: 3
; CHECK-NEXT:   .name:           empty_max_num_workgroups_x_max
; CHECK-NEXT:   .private_segment_fixed_size: 0

; CHECK: - .args:
; CHECK:        .max_flat_workgroup_size: 1024
; CHECK-NEXT:   .max_num_workgroups_x: 1
; CHECK-NEXT:   .max_num_workgroups_z: 3
; CHECK-NEXT:   .name:           empty_max_num_workgroups_y_max
; CHECK-NEXT:   .private_segment_fixed_size: 0

; CHECK: - .args:
; CHECK:        .max_flat_workgroup_size: 1024
; CHECK-NEXT:   .max_num_workgroups_x: 1
; CHECK-NEXT:   .max_num_workgroups_y: 2
; CHECK-NEXT:   .name:           empty_max_num_workgroups_z_max
; CHECK-NEXT:   .private_segment_fixed_size: 0
