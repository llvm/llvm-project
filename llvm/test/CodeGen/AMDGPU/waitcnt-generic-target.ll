; RUN: llc -mtriple=amdgcn -show-mc-encoding < %s | FileCheck %s
; RUN: llc -mtriple=amdgcn -mcpu=generic -show-mc-encoding < %s | FileCheck %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=generic-hsa -show-mc-encoding < %s | FileCheck %s

; Targets without a real CPU use the SI s_waitcnt layout.

; CHECK-LABEL: {{^}}f:
; CHECK: s_waitcnt lgkmcnt(0) ; encoding: [0x7f,0x00,0x8c,0xbf]
; CHECK: s_waitcnt vmcnt(0) ; encoding: [0x70,0x0f,0x8c,0xbf]
define amdgpu_kernel void @f(ptr addrspace(1) %p) {
  %v = load volatile i32, ptr addrspace(1) %p
  store i32 %v, ptr addrspace(1) %p
  ret void
}
