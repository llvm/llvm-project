; RUN: opt -S -mtriple=amdgcn-amd-amdhsa -mcpu=gfx942 -passes=amdgpu-preload-kernel-arguments -amdgpu-kernarg-preload-count=3 < %s | FileCheck %s

define amdgpu_kernel void @global_default(ptr addrspace(1) %a, ptr addrspace(1) %b, ptr addrspace(1) %c) {
; CHECK-LABEL: define amdgpu_kernel void @global_default(
; CHECK-SAME: ptr addrspace(1) inreg %a, ptr addrspace(1) inreg %b, ptr addrspace(1) inreg %c)
  ret void
}

define amdgpu_kernel void @middle_range(ptr addrspace(1) %a, i32 %b, ptr addrspace(1) %c, i32 %d) #0 {
; CHECK-LABEL: define amdgpu_kernel void @middle_range(
; CHECK-SAME: ptr addrspace(1) %a, i32 inreg %b, ptr addrspace(1) inreg %c, i32 %d)
  ret void
}

define amdgpu_kernel void @single_argument(i32 %a, i32 %b, i32 %c) #1 {
; CHECK-LABEL: define amdgpu_kernel void @single_argument(
; CHECK-SAME: i32 %a, i32 %b, i32 inreg %c)
  ret void
}

attributes #0 = { "amdgpu-kernarg-preload-first-arg"="1" "amdgpu-kernarg-preload-last-arg"="2" }
attributes #1 = { "amdgpu-kernarg-preload-first-arg"="2" "amdgpu-kernarg-preload-last-arg"="2" }
