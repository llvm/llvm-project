; RUN: llc -mtriple=amdgcn -mcpu=gfx906 -verify-machineinstrs < %s | FileCheck %s --check-prefixes=GCN,GFX9,GFX906
; RUN: llc -mtriple=amdgcn -mcpu=gfx942 -verify-machineinstrs < %s | FileCheck %s --check-prefixes=GCN,GFX9,GFX942-SDAG
; RUN: llc -mtriple=amdgcn -mcpu=gfx942 -global-isel -verify-machineinstrs < %s | FileCheck %s --check-prefixes=GCN,GFX9,GFX942-GISEL
; RUN: llc -mtriple=amdgcn -mcpu=gfx1011 -verify-machineinstrs < %s | FileCheck %s --check-prefixes=GCN,GFX10
; RUN: llc -mtriple=amdgcn -mcpu=gfx1012 -verify-machineinstrs < %s | FileCheck %s --check-prefixes=GCN,GFX10

declare i32 @llvm.amdgcn.udot2(<2 x i16> %a, <2 x i16> %b, i32 %c, i1 %clamp)
declare i32 @llvm.amdgcn.workitem.id.x()

; GCN-LABEL: {{^}}test_llvm_amdgcn_udot2_clamp:
; GFX9:   v_dot2_u32_u16 v{{[0-9]+}}, s{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}} clamp{{$}}
; GFX10:  v_dot2_u32_u16 v{{[0-9]+}}, s{{[0-9]+}}, s{{[0-9]+}}, v{{[0-9]+}} clamp{{$}}
define amdgpu_kernel void @test_llvm_amdgcn_udot2_clamp(
    ptr addrspace(1) %r,
    ptr addrspace(1) %a,
    ptr addrspace(1) %b,
    ptr addrspace(1) %c) {
entry:
  %a.val = load <2 x i16>, ptr addrspace(1) %a
  %b.val = load <2 x i16>, ptr addrspace(1) %b
  %c.val = load i32, ptr addrspace(1) %c
  %r.val = call i32 @llvm.amdgcn.udot2(<2 x i16> %a.val, <2 x i16> %b.val, i32 %c.val, i1 1)
  store i32 %r.val, ptr addrspace(1) %r
  ret void
}

; GCN-LABEL: {{^}}test_llvm_amdgcn_udot2_no_clamp:
; GFX9:   v_dot2_u32_u16 v{{[0-9]+}}, s{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}{{$}}
; GFX10:  v_dot2_u32_u16 v{{[0-9]+}}, s{{[0-9]+}}, s{{[0-9]+}}, v{{[0-9]+}}{{$}}
define amdgpu_kernel void @test_llvm_amdgcn_udot2_no_clamp(
    ptr addrspace(1) %r,
    ptr addrspace(1) %a,
    ptr addrspace(1) %b,
    ptr addrspace(1) %c) {
entry:
  %a.val = load <2 x i16>, ptr addrspace(1) %a
  %b.val = load <2 x i16>, ptr addrspace(1) %b
  %c.val = load i32, ptr addrspace(1) %c
  %r.val = call i32 @llvm.amdgcn.udot2(<2 x i16> %a.val, <2 x i16> %b.val, i32 %c.val, i1 0)
  store i32 %r.val, ptr addrspace(1) %r
  ret void
}

; GCN-LABEL: {{^}}test_llvm_amdgcn_udot2_op_sel:
; GFX906: v_dot2_u32_u16 v{{[0-9]+}}, 1, v{{[0-9]+}}, s{{[0-9]+}} op_sel:[0,1,0] op_sel_hi:[0,0,1]{{$}}
; GFX942-SDAG: s_mov_b32 [[K:s[0-9]+]], 0x10001
; GFX942-SDAG: v_dot2_u32_u16 v{{[0-9]+}}, [[K]], v{{[0-9]+}}, v{{[0-9]+}}{{$}}
; GFX942-GISEL: v_mov_b32_e32 [[K:v[0-9]+]], 0x10001
; GFX942-GISEL: v_dot2_u32_u16 v{{[0-9]+}}, [[K]], v{{[0-9]+}}, s{{[0-9]+}}{{$}}
; GFX10:  v_dot2_u32_u16 v{{[0-9]+}}, 1, v{{[0-9]+}}, s{{[0-9]+}} op_sel:[0,1,0] op_sel_hi:[0,0,1]{{$}}
define amdgpu_kernel void @test_llvm_amdgcn_udot2_op_sel(
    ptr addrspace(1) %r,
    ptr addrspace(1) %b,
    i32 %c) {
entry:
  %id = tail call i32 @llvm.amdgcn.workitem.id.x()
  %b.gep = getelementptr inbounds <2 x i16>, ptr addrspace(1) %b, i32 %id
  %b.val = load <2 x i16>, ptr addrspace(1) %b.gep
  %b.elt0 = extractelement <2 x i16> %b.val, i32 0
  %b.elt1 = extractelement <2 x i16> %b.val, i32 1
  %b0 = insertelement <2 x i16> undef, i16 %b.elt1, i32 0
  %b1 = insertelement <2 x i16> %b0, i16 %b.elt0, i32 1
  %r.val = call i32 @llvm.amdgcn.udot2(<2 x i16> <i16 1, i16 1>, <2 x i16> %b1, i32 %c, i1 0)
  store i32 %r.val, ptr addrspace(1) %r
  ret void
}
