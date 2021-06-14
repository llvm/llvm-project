; RUN: llc -march=amdgcn -mcpu=gfx1100 -verify-machineinstrs < %s | FileCheck %s --check-prefixes=GCN,GFX11

declare float @llvm.amdgcn.fdot2.f32.bf16(<2 x i16> %a, <2 x i16> %b, float %c, i1 %clamp)

; GCN-LABEL: test_llvm_amdgcn_fdot2_f32_bf16_clamp:
; GFX11:     v_dot2_f32_bf16 v{{[0-9]+}}, s{{[0-9]+}}, s{{[0-9]+}}, v{{[0-9]+}} clamp{{$}}
define amdgpu_kernel void @test_llvm_amdgcn_fdot2_f32_bf16_clamp(
    float addrspace(1)* %r,
    <2 x i16> addrspace(1)* %a,
    <2 x i16> addrspace(1)* %b,
    float addrspace(1)* %c) {
entry:
  %a.val = load <2 x i16>, <2 x i16> addrspace(1)* %a
  %b.val = load <2 x i16>, <2 x i16> addrspace(1)* %b
  %c.val = load float, float addrspace(1)* %c
  %r.val = call float @llvm.amdgcn.fdot2.f32.bf16(<2 x i16> %a.val, <2 x i16> %b.val, float %c.val, i1 1)
  store float %r.val, float addrspace(1)* %r
  ret void
}

; GCN-LABEL: test_llvm_amdgcn_fdot2_f32_bf16_no_clamp:
; GFX11:     v_dot2_f32_bf16 v{{[0-9]+}}, s{{[0-9]+}}, s{{[0-9]+}}, v{{[0-9]+}}{{$}}
define amdgpu_kernel void @test_llvm_amdgcn_fdot2_f32_bf16_no_clamp(
    float addrspace(1)* %r,
    <2 x i16> addrspace(1)* %a,
    <2 x i16> addrspace(1)* %b,
    float addrspace(1)* %c) {
entry:
  %a.val = load <2 x i16>, <2 x i16> addrspace(1)* %a
  %b.val = load <2 x i16>, <2 x i16> addrspace(1)* %b
  %c.val = load float, float addrspace(1)* %c
  %r.val = call float @llvm.amdgcn.fdot2.f32.bf16(<2 x i16> %a.val, <2 x i16> %b.val, float %c.val, i1 0)
  store float %r.val, float addrspace(1)* %r
  ret void
}
