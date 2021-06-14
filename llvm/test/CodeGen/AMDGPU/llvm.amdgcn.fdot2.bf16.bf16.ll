; RUN: llc -march=amdgcn -mcpu=gfx1100 -verify-machineinstrs < %s | FileCheck %s --check-prefixes=GCN,GFX11

declare i16 @llvm.amdgcn.fdot2.bf16.bf16(<2 x i16> %a, <2 x i16> %b, i16 %c)

; GCN-LABEL: test_llvm_amdgcn_fdot2_f16_f16:
; GFX11:     v_dot2_bf16_bf16 v{{[0-9]+}}, s{{[0-9]+}}, s{{[0-9]+}}, v{{[0-9]+}}{{$}}
define amdgpu_kernel void @test_llvm_amdgcn_fdot2_f16_f16(
    i16 addrspace(1)* %r,
    <2 x i16> addrspace(1)* %a,
    <2 x i16> addrspace(1)* %b,
    i16 addrspace(1)* %c) {
entry:
  %a.val = load <2 x i16>, <2 x i16> addrspace(1)* %a
  %b.val = load <2 x i16>, <2 x i16> addrspace(1)* %b
  %c.val = load i16, i16 addrspace(1)* %c
  %r.val = call i16 @llvm.amdgcn.fdot2.bf16.bf16(<2 x i16> %a.val, <2 x i16> %b.val, i16 %c.val)
  store i16 %r.val, i16 addrspace(1)* %r
  ret void
}
