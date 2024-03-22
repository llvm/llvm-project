; RUN: not llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 < %s 2>&1 | FileCheck --check-prefix=ERROR %s

; ERROR: error: can't parse integer attribute -1 in amdgpu-max-num-workgroups
define amdgpu_kernel void @empty_max_num_workgroups_neg_num1() "amdgpu-max-num-workgroups"="-1,2,3" {
entry:
  ret void
}

; ERROR: error: can't parse integer attribute -2 in amdgpu-max-num-workgroups
define amdgpu_kernel void @empty_max_num_workgroups_neg_num2() "amdgpu-max-num-workgroups"="1,-2,3" {
entry:
  ret void
}

; ERROR: error: can't parse integer attribute -3 in amdgpu-max-num-workgroups
define amdgpu_kernel void @empty_max_num_workgroups_neg_num3() "amdgpu-max-num-workgroups"="1,2,-3" {
entry:
  ret void
}

; ERROR: error: can't parse integer attribute 1.0 in amdgpu-max-num-workgroups
define amdgpu_kernel void @empty_max_num_workgroups_non_int1() "amdgpu-max-num-workgroups"="1.0,2,3" {
entry:
  ret void
}

; ERROR: error: can't parse integer attribute 2.0 in amdgpu-max-num-workgroups
define amdgpu_kernel void @empty_max_num_workgroups_non_int2() "amdgpu-max-num-workgroups"="1,2.0,3" {
entry:
  ret void
}

; ERROR: error: can't parse integer attribute 3.0 in amdgpu-max-num-workgroups
define amdgpu_kernel void @empty_max_num_workgroups_non_int3() "amdgpu-max-num-workgroups"="1,2,3.0" {
entry:
  ret void
}

; ERROR: error: can't parse integer attribute 10000000000 in amdgpu-max-num-workgroups
define amdgpu_kernel void @empty_max_num_workgroups_too_large() "amdgpu-max-num-workgroups"="10000000000,2,3" {
entry:
  ret void
}

; ERROR: error: attribute amdgpu-max-num-workgroups has incorrect number of integers; expected 3
define amdgpu_kernel void @empty_max_num_workgroups_1_arg() "amdgpu-max-num-workgroups"="1" {
entry:
  ret void
}

; ERROR: error: attribute amdgpu-max-num-workgroups has incorrect number of integers; expected 3
define amdgpu_kernel void @empty_max_num_workgroups_2_args() "amdgpu-max-num-workgroups"="1,2" {
entry:
  ret void
}

; ERROR: error: attribute amdgpu-max-num-workgroups has incorrect number of integers; expected 3
define amdgpu_kernel void @empty_max_num_workgroups_4_args() "amdgpu-max-num-workgroups"="1,2,3,4" {
entry:
  ret void
}
