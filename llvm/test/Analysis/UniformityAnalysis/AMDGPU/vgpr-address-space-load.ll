; RUN: opt -mtriple amdgpu12.00-- -passes='print<uniformity>' -disable-output %s 2>&1 | FileCheck %s

; Each lane reads its own registers, so a load from the VGPR address space is
; divergent even through a uniform pointer. A global load through a uniform
; pointer stays uniform.

; CHECK-LABEL: for function 'load_uniform_ptr':
; CHECK: DIVERGENT: %vgpr = load i32, ptr addrspace(13) %p
; CHECK-NOT: DIVERGENT
; CHECK: TERMINATORS
define amdgpu_kernel void @load_uniform_ptr(ptr addrspace(13) %p, ptr addrspace(1) %q) {
  %vgpr = load i32, ptr addrspace(13) %p
  %global = load i32, ptr addrspace(1) %q
  ret void
}
