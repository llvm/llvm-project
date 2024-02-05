; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -verify-machineinstrs < %s | FileCheck %s

; Attribute not specified.
; CHECK-LABEL: {{^}}empty_no_attribute:
define amdgpu_kernel void @empty_no_attribute() {
entry:
  ret void
}

; Ignore if number of work groups for x dimension is 0.
; CHECK-LABEL: {{^}}empty_min_num_work_groups_x0:
define amdgpu_kernel void @empty_min_num_work_groups_x0() #0 {
entry:
  ret void
}
attributes #0 = {"amdgpu-min-num-work-groups"="0,2,3"}

; Ignore if number of work groups for x dimension is 0.
; CHECK-LABEL: {{^}}empty_max_num_work_groups_x0:
define amdgpu_kernel void @empty_max_num_work_groups_x0() #1 {
entry:
  ret void
}
attributes #1 = {"amdgpu-max-num-work-groups"="0,2,3"}

; Ignore if number of work groups for y dimension is 0.
; CHECK-LABEL: {{^}}empty_min_num_work_groups_y0:
define amdgpu_kernel void @empty_min_num_work_groups_y0() #2 {
entry:
  ret void
}
attributes #2 = {"amdgpu-min-num-work-groups"="1,0,3"}

; Ignore if number of work groups for y dimension is 0.
; CHECK-LABEL: {{^}}empty_max_num_work_groups_y0:
define amdgpu_kernel void @empty_max_num_work_groups_y0() #3 {
entry:
  ret void
}
attributes #3 = {"amdgpu-max-num-work-groups"="1,0,3"}

; Ignore if number of work groups for z dimension is 0.
; CHECK-LABEL: {{^}}empty_min_num_work_groups_z0:
define amdgpu_kernel void @empty_min_num_work_groups_z0() #4 {
entry:
  ret void
}
attributes #4 = {"amdgpu-min-num-work-groups"="1,2,0"}

; Ignore if number of work groups for z dimension is 0.
; CHECK-LABEL: {{^}}empty_max_num_work_groups_z0:
define amdgpu_kernel void @empty_max_num_work_groups_z0() #5 {
entry:
  ret void
}
attributes #5 = {"amdgpu-max-num-work-groups"="1,2,0"}



; CHECK-LABEL: {{^}}empty_min_num_work_groups_1_2_3:
define amdgpu_kernel void @empty_min_num_work_groups_1_2_3() #20 {
entry:
  ret void
}
attributes #20 = {"amdgpu-min-num-work-groups"="1,2,3"}

; CHECK-LABEL: {{^}}empty_max_num_work_groups_1_2_3:
define amdgpu_kernel void @empty_max_num_work_groups_1_2_3() #21 {
entry:
  ret void
}
attributes #21 = {"amdgpu-max-num-work-groups"="1,2,3"}

; CHECK-LABEL: {{^}}empty_min_num_work_groups_1024_1024_1024:
define amdgpu_kernel void @empty_min_num_work_groups_1024_1024_1024() #22 {
entry:
  ret void
}
attributes #22 = {"amdgpu-min-num-work-groups"="1024,1024,1024"}

; CHECK-LABEL: {{^}}empty_max_num_work_groups_1024_1024_1024:
define amdgpu_kernel void @empty_max_num_work_groups_1024_1024_1024() #23 {
entry:
  ret void
}
attributes #23 = {"amdgpu-max-num-work-groups"="1024,1024,1024"}


; CHECK-LABEL: {{^}}empty_min_max_num_work_groups_bad_min:
define amdgpu_kernel void @empty_min_max_num_work_groups_bad_min() #30 {
entry:
  ret void
}
attributes #30 = {"amdgpu-min-num-work-groups"="0,2,3" "amdgpu-max-num-work-groups"="1,2,3"}

; CHECK-LABEL: {{^}}empty_min_max_num_work_groups_bad_max:
define amdgpu_kernel void @empty_min_max_num_work_groups_bad_max() #31 {
entry:
  ret void
}
attributes #31 = {"amdgpu-min-num-work-groups"="1,2,3" "amdgpu-max-num-work-groups"="0,2,3"}


; CHECK-LABEL: {{^}}empty_min_max_num_work_groups_bad_x:
define amdgpu_kernel void @empty_min_max_num_work_groups_bad_x() #40 {
entry:
  ret void
}
attributes #40 = {"amdgpu-min-num-work-groups"="2,3,4" "amdgpu-max-num-work-groups"="1,3,4"}

; CHECK-LABEL: {{^}}empty_min_max_num_work_groups_bad_y:
define amdgpu_kernel void @empty_min_max_num_work_groups_bad_y() #41 {
entry:
  ret void
}
attributes #41 = {"amdgpu-min-num-work-groups"="2,3,4" "amdgpu-max-num-work-groups"="2,1,4"}

; CHECK-LABEL: {{^}}empty_min_max_num_work_groups_bad_z:
define amdgpu_kernel void @empty_min_max_num_work_groups_bad_z() #42 {
entry:
  ret void
}
attributes #42 = {"amdgpu-min-num-work-groups"="2,3,4" "amdgpu-max-num-work-groups"="2,3,1"}


; CHECK-LABEL: {{^}}empty_min_max_num_work_groups_equal1:
define amdgpu_kernel void @empty_min_max_num_work_groups_equal1() #50 {
entry:
  ret void
}
attributes #50 = {"amdgpu-min-num-work-groups"="2,3,4" "amdgpu-max-num-work-groups"="2,3,4"}


; CHECK-LABEL: {{^}}empty_min_max_num_work_groups_greater_or_equal1:
define amdgpu_kernel void @empty_min_max_num_work_groups_greater_or_equal1() #60 {
entry:
  ret void
}
attributes #60 = {"amdgpu-min-num-work-groups"="2,3,4" "amdgpu-max-num-work-groups"="2,30,40"}

; CHECK-LABEL: {{^}}empty_min_max_num_work_groups_greater_or_equal2:
define amdgpu_kernel void @empty_min_max_num_work_groups_greater_or_equal2() #61 {
entry:
  ret void
}
attributes #61 = {"amdgpu-min-num-work-groups"="2,3,4" "amdgpu-max-num-work-groups"="20,3,40"}

; CHECK-LABEL: {{^}}empty_min_max_num_work_groups_greater_or_equal3:
define amdgpu_kernel void @empty_min_max_num_work_groups_greater_or_equal3() #62 {
entry:
  ret void
}
attributes #62 = {"amdgpu-min-num-work-groups"="2,3,4" "amdgpu-max-num-work-groups"="20,30,4"}


; CHECK-LABEL: {{^}}empty_min_max_num_work_groups_greater1:
define amdgpu_kernel void @empty_min_max_num_work_groups_greater1() #62 {
entry:
  ret void
}
attributes #62 = {"amdgpu-min-num-work-groups"="2,3,4" "amdgpu-max-num-work-groups"="20,30,40"}


; CHECK: .amdgpu_metadata
; CHECK:        .args:
; CHECK:        .max_flat_workgroup_size: 1024
; CHECK-NEXT:   .name:           empty_no_attribute

; CHECK:        .args:
; CHECK:        .max_flat_workgroup_size: 1024
; CHECK-NEXT:   .name:           empty_min_num_work_groups_x0

; CHECK:        .args:
; CHECK:        .max_flat_workgroup_size: 1024
; CHECK-NEXT:   .name:           empty_max_num_work_groups_x0

; CHECK:        .args:
; CHECK:        .max_flat_workgroup_size: 1024
; CHECK-NEXT:   .name:           empty_min_num_work_groups_y0

; CHECK:        .args:
; CHECK:        .max_flat_workgroup_size: 1024
; CHECK-NEXT:   .name:           empty_max_num_work_groups_y0

; CHECK:        .args:
; CHECK:        .max_flat_workgroup_size: 1024
; CHECK-NEXT:   .name:           empty_min_num_work_groups_z0

; CHECK:        .args:
; CHECK:        .max_flat_workgroup_size: 1024
; CHECK-NEXT:   .name:           empty_max_num_work_groups_z0

; CHECK:        .args:
; CHECK:        .max_flat_workgroup_size: 1024
; CHECK-NEXT:   .min_num_work_groups_x: 1
; CHECK-NEXT:   .min_num_work_groups_y: 2
; CHECK-NEXT:   .min_num_work_groups_z: 3
; CHECK:        .name:           empty_min_num_work_groups_1_2_3

; CHECK:        .args:
; CHECK:        .max_flat_workgroup_size: 1024
; CHECK-NEXT:   .max_num_work_groups_x: 1
; CHECK-NEXT:   .max_num_work_groups_y: 2
; CHECK-NEXT:   .max_num_work_groups_z: 3
; CHECK-NEXT:   .name:           empty_max_num_work_groups_1_2_3

; CHECK:        .args:
; CHECK:        .max_flat_workgroup_size: 1024
; CHECK-NEXT:   .min_num_work_groups_x: 1024
; CHECK-NEXT:   .min_num_work_groups_y: 1024
; CHECK-NEXT:   .min_num_work_groups_z: 1024
; CHECK-NEXT:   .name:           empty_min_num_work_groups_1024_1024_1024

; CHECK:        .args:
; CHECK:        .max_flat_workgroup_size: 1024
; CHECK-NEXT:   .max_num_work_groups_x: 1024
; CHECK-NEXT:   .max_num_work_groups_y: 1024
; CHECK-NEXT:   .max_num_work_groups_z: 1024
; CHECK-NEXT:   .name:           empty_max_num_work_groups_1024_1024_1024

; CHECK:        .args:
; CHECK:        .max_flat_workgroup_size: 1024
; CHECK-NEXT:   .max_num_work_groups_x: 1
; CHECK-NEXT:   .max_num_work_groups_y: 2
; CHECK-NEXT:   .max_num_work_groups_z: 3
; CHECK-NEXT:   .name:           empty_min_max_num_work_groups_bad_min

; CHECK:        .args:
; CHECK:        .max_flat_workgroup_size: 1024
; CHECK-NEXT:   .min_num_work_groups_x: 1
; CHECK-NEXT:   .min_num_work_groups_y: 2
; CHECK-NEXT:   .min_num_work_groups_z: 3
; CHECK-NEXT:   .name:           empty_min_max_num_work_groups_bad_max

; CHECK:        .args:
; CHECK:        .max_flat_workgroup_size: 1024
; CHECK-NEXT:   .name:           empty_min_max_num_work_groups_bad_x

; CHECK:        .args:
; CHECK:        .max_flat_workgroup_size: 1024
; CHECK-NEXT:   .name:           empty_min_max_num_work_groups_bad_y

; CHECK:        .args:
; CHECK:        .max_flat_workgroup_size: 1024
; CHECK-NEXT:   .name:           empty_min_max_num_work_groups_bad_z

; CHECK:        .args:
; CHECK:        .max_flat_workgroup_size: 1024
; CHECK-NEXT:   .max_num_work_groups_x: 2
; CHECK-NEXT:   .max_num_work_groups_y: 3
; CHECK-NEXT:   .max_num_work_groups_z: 4
; CHECK-NEXT:   .min_num_work_groups_x: 2
; CHECK-NEXT:   .min_num_work_groups_y: 3
; CHECK-NEXT:   .min_num_work_groups_z: 4
; CHECK-NEXT:   .name:           empty_min_max_num_work_groups_equal1

; CHECK:        .args:
; CHECK:        .max_flat_workgroup_size: 1024
; CHECK-NEXT:   .max_num_work_groups_x: 2
; CHECK-NEXT:   .max_num_work_groups_y: 30
; CHECK-NEXT:   .max_num_work_groups_z: 40
; CHECK-NEXT:   .min_num_work_groups_x: 2
; CHECK-NEXT:   .min_num_work_groups_y: 3
; CHECK-NEXT:   .min_num_work_groups_z: 4
; CHECK-NEXT:   .name:           empty_min_max_num_work_groups_greater_or_equal1

; CHECK:        .args:
; CHECK:        .max_flat_workgroup_size: 1024
; CHECK-NEXT:   .max_num_work_groups_x: 20
; CHECK-NEXT:   .max_num_work_groups_y: 3
; CHECK-NEXT:   .max_num_work_groups_z: 40
; CHECK-NEXT:   .min_num_work_groups_x: 2
; CHECK-NEXT:   .min_num_work_groups_y: 3
; CHECK-NEXT:   .min_num_work_groups_z: 4
; CHECK-NEXT:   .name:           empty_min_max_num_work_groups_greater_or_equal2

; CHECK:        .args:
; CHECK:        .max_flat_workgroup_size: 1024
; CHECK-NEXT:   .max_num_work_groups_x: 20
; CHECK-NEXT:   .max_num_work_groups_y: 30
; CHECK-NEXT:   .max_num_work_groups_z: 4
; CHECK-NEXT:   .min_num_work_groups_x: 2
; CHECK-NEXT:   .min_num_work_groups_y: 3
; CHECK-NEXT:   .min_num_work_groups_z: 4
; CHECK-NEXT:   .name:           empty_min_max_num_work_groups_greater_or_equal3

; CHECK:        .args:
; CHECK:        .max_flat_workgroup_size: 1024
; CHECK-NEXT:   .max_num_work_groups_x: 20
; CHECK-NEXT:   .max_num_work_groups_y: 30
; CHECK-NEXT:   .max_num_work_groups_z: 40
; CHECK-NEXT:   .min_num_work_groups_x: 2
; CHECK-NEXT:   .min_num_work_groups_y: 3
; CHECK-NEXT:   .min_num_work_groups_z: 4
; CHECK-NEXT:   .name:           empty_min_max_num_work_groups_greater1
