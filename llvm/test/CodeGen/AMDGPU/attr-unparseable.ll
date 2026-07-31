; RUN: not llc -mtriple=amdgpu8.03--amdhsa < %s 2>&1 | FileCheck %s

; CHECK: can't parse first integer attribute amdgpu-flat-work-group-size
define amdgpu_kernel void @unparseable_pair_0() #3 {
entry:
  ret void
}
attributes #3 = {"amdgpu-flat-work-group-size"}

; CHECK: can't parse first integer attribute amdgpu-flat-work-group-size
define amdgpu_kernel void @unparseable_pair_1() #4 {
entry:
  ret void
}
attributes #4 = {"amdgpu-flat-work-group-size"="k"}

; CHECK: can't parse second integer attribute amdgpu-flat-work-group-size
define amdgpu_kernel void @unparseable_pair_2() #5 {
entry:
  ret void
}
attributes #5 = {"amdgpu-flat-work-group-size"="1"}

; CHECK: can't parse second integer attribute amdgpu-flat-work-group-size
define amdgpu_kernel void @unparseable_pair_3() #6 {
entry:
  ret void
}
attributes #6 = {"amdgpu-flat-work-group-size"="1,k"}

; CHECK: can't parse second integer attribute amdgpu-flat-work-group-size
define amdgpu_kernel void @unparseable_pair_4() #7 {
entry:
  ret void
}
attributes #7 = {"amdgpu-flat-work-group-size"="1,2,3"}
