; RUN: not --crash llc -global-isel -mtriple=amdgcn-amd-amdpal -mcpu=gfx900 < %s
; RUN: not --crash llc -global-isel -mtriple=amdgcn-amd-amdpal -mcpu=fiji < %s
; RUN: not --crash llc -global-isel -mtriple=amdgcn-amd-amdpal -mcpu=tahiti < %s

define i96 @zextload_global_i32_to_i96(i32 addrspace(1)* %ptr) {
  %load = load i32, i32 addrspace(1)* %ptr
  %ext = zext i32 %load to i96
  ret i96 %ext
}
