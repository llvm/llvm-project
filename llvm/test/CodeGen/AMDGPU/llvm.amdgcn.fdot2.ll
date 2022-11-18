; RUN: llc -march=amdgcn -mcpu=gfx906 -verify-machineinstrs < %s | FileCheck %s --check-prefixes=GCN,GFX9,GFX906
; RUN: llc -march=amdgcn -mcpu=gfx940 -verify-machineinstrs < %s | FileCheck %s --check-prefixes=GCN,GFX9,GFX940
; RUN: llc -march=amdgcn -mcpu=gfx1011 -verify-machineinstrs < %s | FileCheck %s --check-prefixes=GCN,GFX10
; RUN: llc -march=amdgcn -mcpu=gfx1012 -verify-machineinstrs < %s | FileCheck %s --check-prefixes=GCN,GFX10
; RUN: llc -march=amdgcn -mcpu=gfx1100 -amdgpu-enable-vopd=0 -verify-machineinstrs < %s | FileCheck %s --check-prefixes=GCN,GFX10

declare float @llvm.amdgcn.fdot2(<2 x half> %a, <2 x half> %b, float %c, i1 %clamp)

; GCN-LABEL: {{^}}test_llvm_amdgcn_fdot2_clamp
; GFX9:   v_dot2_f32_f16 v{{[0-9]+}}, s{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}} clamp{{$}}
; GFX10:  v_dot2_f32_f16 v{{[0-9]+}}, s{{[0-9]+}}, s{{[0-9]+}}, v{{[0-9]+}} clamp{{$}}
define amdgpu_kernel void @test_llvm_amdgcn_fdot2_clamp(
    float addrspace(1)* %r,
    <2 x half> addrspace(1)* %a,
    <2 x half> addrspace(1)* %b,
    float addrspace(1)* %c) {
entry:
  %a.val = load <2 x half>, <2 x half> addrspace(1)* %a
  %b.val = load <2 x half>, <2 x half> addrspace(1)* %b
  %c.val = load float, float addrspace(1)* %c
  %r.val = call float @llvm.amdgcn.fdot2(<2 x half> %a.val, <2 x half> %b.val, float %c.val, i1 1)
  store float %r.val, float addrspace(1)* %r
  ret void
}

; GCN-LABEL: {{^}}test_llvm_amdgcn_fdot2_no_clamp
; GFX906: v_dot2_f32_f16 v{{[0-9]+}}, s{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}{{$}}
; GFX940: v_dot2c_f32_f16_e32 v{{[0-9]+}}, s{{[0-9]+}}, v{{[0-9]+}}{{$}}
; GFX10:  {{v_dot2c_f32_f16_e32|v_dot2acc_f32_f16}} v{{[0-9]+}}, s{{[0-9]+}}, v{{[0-9]+}}{{$}}
define amdgpu_kernel void @test_llvm_amdgcn_fdot2_no_clamp(
    float addrspace(1)* %r,
    <2 x half> addrspace(1)* %a,
    <2 x half> addrspace(1)* %b,
    float addrspace(1)* %c) {
entry:
  %a.val = load <2 x half>, <2 x half> addrspace(1)* %a
  %b.val = load <2 x half>, <2 x half> addrspace(1)* %b
  %c.val = load float, float addrspace(1)* %c
  %r.val = call float @llvm.amdgcn.fdot2(<2 x half> %a.val, <2 x half> %b.val, float %c.val, i1 0)
  store float %r.val, float addrspace(1)* %r
  ret void
}

; GFX9-LABEL: {{^}}fdot2_inline_literal
; GFX906: v_dot2_f32_f16 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}, 1.0
; GFX940: v_dot2c_f32_f16_e32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}{{$}}
define float @fdot2_inline_literal(<2 x half> %a, <2 x half> %b) {
  %ret = tail call float @llvm.amdgcn.fdot2(<2 x half> %a, <2 x half> %b, float 1.0, i1 false)
  ret float %ret
}
