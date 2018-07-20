; RUN: llc -march=amdgcn -mcpu=gfx906 -verify-machineinstrs < %s | FileCheck %s --check-prefix=GFX906

declare float @llvm.amdgcn.fdot2(<2 x half> %a, <2 x half> %b, float %c)

; GFX906-LABEL: {{^}}test_llvm_amdgcn_fdot2
; GFX906: v_dot2_f32_f16
define amdgpu_kernel void @test_llvm_amdgcn_fdot2(
    float addrspace(1)* %r,
    <2 x half> addrspace(1)* %a,
    <2 x half> addrspace(1)* %b,
    float addrspace(1)* %c) {
entry:
  %a.val = load <2 x half>, <2 x half> addrspace(1)* %a
  %b.val = load <2 x half>, <2 x half> addrspace(1)* %b
  %c.val = load float, float addrspace(1)* %c
  %r.val = call float @llvm.amdgcn.fdot2(<2 x half> %a.val, <2 x half> %b.val, float %c.val)
  store float %r.val, float addrspace(1)* %r
  ret void
}
