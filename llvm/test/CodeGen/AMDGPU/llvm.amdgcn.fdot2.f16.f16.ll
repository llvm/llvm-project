; RUN: llc -march=amdgcn -mcpu=gfx1100 -verify-machineinstrs < %s | FileCheck %s --check-prefixes=GCN,GFX11

declare half @llvm.amdgcn.fdot2.f16.f16(<2 x half> %a, <2 x half> %b, half %c)

; GCN-LABEL: test_llvm_amdgcn_fdot2_f16_f16:
; GFX11:     v_dot2_f16_f16 v{{[0-9]+}}, s{{[0-9]+}}, s{{[0-9]+}}, v{{[0-9]+}}{{$}}
define amdgpu_kernel void @test_llvm_amdgcn_fdot2_f16_f16(
    half addrspace(1)* %r,
    <2 x half> addrspace(1)* %a,
    <2 x half> addrspace(1)* %b,
    half addrspace(1)* %c) {
entry:
  %a.val = load <2 x half>, <2 x half> addrspace(1)* %a
  %b.val = load <2 x half>, <2 x half> addrspace(1)* %b
  %c.val = load half, half addrspace(1)* %c
  %r.val = call half @llvm.amdgcn.fdot2.f16.f16(<2 x half> %a.val, <2 x half> %b.val, half %c.val)
  store half %r.val, half addrspace(1)* %r
  ret void
}
