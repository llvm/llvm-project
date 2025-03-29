; REQUIRES: amdgpu-registered-target
; RUN: opt -S -mtriple=amdgcn-amd-amdhsa -passes=hipstdpar-select-accelerator-code \
; RUN: %s | FileCheck %s

; CHECK: @var = extern_weak addrspace(1) externally_initialized global i32, align 4
@var = addrspace(1) global i32 0, align 4

define amdgpu_kernel void @kernel() {
entry:
  store i32 1, ptr addrspace(1) @var, align 4
  ret void
}
