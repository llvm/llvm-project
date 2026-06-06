; RUN: not llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 < %s 2>&1 | FileCheck --check-prefix=ERROR %s

; ERROR: error: can't parse integer attribute -1 in amdgpu-max-num-workgroups
define amdgpu_kernel void @empty_max_num_workgroups_neg_num1() #21 {
entry:
  ret void
}
attributes #21 = {"amdgpu-max-num-workgroups"="-1,2,3"}

; ERROR: error: can't parse integer attribute -2 in amdgpu-max-num-workgroups
define amdgpu_kernel void @empty_max_num_workgroups_neg_num2() #22 {
entry:
  ret void
}
attributes #22 = {"amdgpu-max-num-workgroups"="1,-2,3"}

; ERROR: error: can't parse integer attribute -3 in amdgpu-max-num-workgroups
define amdgpu_kernel void @empty_max_num_workgroups_neg_num3() #23 {
entry:
  ret void
}
attributes #23 = {"amdgpu-max-num-workgroups"="1,2,-3"}

; ERROR: error: can't parse integer attribute 1.0 in amdgpu-max-num-workgroups
define amdgpu_kernel void @empty_max_num_workgroups_non_int1() #31 {
entry:
  ret void
}
attributes #31 = {"amdgpu-max-num-workgroups"="1.0,2,3"}

; ERROR: error: can't parse integer attribute 2.0 in amdgpu-max-num-workgroups
define amdgpu_kernel void @empty_max_num_workgroups_non_int2() #32 {
entry:
  ret void
}
attributes #32 = {"amdgpu-max-num-workgroups"="1,2.0,3"}

; ERROR: error: can't parse integer attribute 3.0 in amdgpu-max-num-workgroups
define amdgpu_kernel void @empty_max_num_workgroups_non_int3() #33 {
entry:
  ret void
}
attributes #33 = {"amdgpu-max-num-workgroups"="1,2,3.0"}

; ERROR: error: can't parse integer attribute 10000000000 in amdgpu-max-num-workgroups
define amdgpu_kernel void @empty_max_num_workgroups_too_large() #41 {
entry:
  ret void
}
attributes #41 = {"amdgpu-max-num-workgroups"="10000000000,2,3"}

; ERROR: error: attribute amdgpu-max-num-workgroups has incorrect number of integers; expected 3
define amdgpu_kernel void @empty_max_num_workgroups_1_arg() #51 {
entry:
  ret void
}
attributes #51 = {"amdgpu-max-num-workgroups"="1"}

; ERROR: error: attribute amdgpu-max-num-workgroups has incorrect number of integers; expected 3
define amdgpu_kernel void @empty_max_num_workgroups_2_args() #52 {
entry:
  ret void
}
attributes #52 = {"amdgpu-max-num-workgroups"="1,2"}

; ERROR: error: attribute amdgpu-max-num-workgroups has incorrect number of integers; expected 3
define amdgpu_kernel void @empty_max_num_workgroups_4_args() #53 {
entry:
  ret void
}
attributes #53 = {"amdgpu-max-num-workgroups"="1,2,3,4"}
