; RUN: not llc -mtriple=amdgcn--amdhsa -mcpu=fiji -verify-machineinstrs < %s 2>&1 | FileCheck %s

; CHECK: cannot parse integer attribute amdgpu-num-sgpr
define amdgpu_kernel void @unparseable_single_0() "amdgpu-num-sgpr" {
entry:
  ret void
}

; CHECK: cannot parse integer attribute amdgpu-num-sgpr
define amdgpu_kernel void @unparseable_single_1() "amdgpu-num-sgpr"="k" {
entry:
  ret void
}

; CHECK: cannot parse integer attribute amdgpu-num-sgpr
define amdgpu_kernel void @unparseable_single_2() "amdgpu-num-sgpr"="1,2" {
entry:
  ret void
}

; CHECK: can't parse first integer attribute amdgpu-flat-work-group-size
define amdgpu_kernel void @unparseable_pair_0() "amdgpu-flat-work-group-size" {
entry:
  ret void
}

; CHECK: can't parse first integer attribute amdgpu-flat-work-group-size
define amdgpu_kernel void @unparseable_pair_1() "amdgpu-flat-work-group-size"="k" {
entry:
  ret void
}

; CHECK: can't parse second integer attribute amdgpu-flat-work-group-size
define amdgpu_kernel void @unparseable_pair_2() "amdgpu-flat-work-group-size"="1" {
entry:
  ret void
}

; CHECK: can't parse second integer attribute amdgpu-flat-work-group-size
define amdgpu_kernel void @unparseable_pair_3() "amdgpu-flat-work-group-size"="1,k" {
entry:
  ret void
}

; CHECK: can't parse second integer attribute amdgpu-flat-work-group-size
define amdgpu_kernel void @unparseable_pair_4() "amdgpu-flat-work-group-size"="1,2,3" {
entry:
  ret void
}
