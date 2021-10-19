; RUN: llc -march=amdgcn -mcpu=gfx906 -verify-machineinstrs < %s | FileCheck %s --check-prefixes=GCN,GFX906
; RUN: llc -march=amdgcn -mcpu=gfx1011 -verify-machineinstrs < %s | FileCheck %s --check-prefixes=GCN,GFX10
; RUN: llc -march=amdgcn -mcpu=gfx1012 -verify-machineinstrs < %s | FileCheck %s --check-prefixes=GCN,GFX10

declare i32 @llvm.amdgcn.udot2(<2 x i16> %a, <2 x i16> %b, i32 %c, i1 %clamp)
declare i32 @llvm.amdgcn.workitem.id.x()

; GCN-LABEL: {{^}}test_llvm_amdgcn_udot2_clamp:
; GFX906: v_dot2_u32_u16 v{{[0-9]+}}, s{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}} clamp{{$}}
; GFX10:  v_dot2_u32_u16 v{{[0-9]+}}, s{{[0-9]+}}, s{{[0-9]+}}, v{{[0-9]+}} clamp{{$}}
define amdgpu_kernel void @test_llvm_amdgcn_udot2_clamp(
    i32 addrspace(1)* %r,
    <2 x i16> addrspace(1)* %a,
    <2 x i16> addrspace(1)* %b,
    i32 addrspace(1)* %c) {
entry:
  %a.val = load <2 x i16>, <2 x i16> addrspace(1)* %a
  %b.val = load <2 x i16>, <2 x i16> addrspace(1)* %b
  %c.val = load i32, i32 addrspace(1)* %c
  %r.val = call i32 @llvm.amdgcn.udot2(<2 x i16> %a.val, <2 x i16> %b.val, i32 %c.val, i1 1)
  store i32 %r.val, i32 addrspace(1)* %r
  ret void
}

; GCN-LABEL: {{^}}test_llvm_amdgcn_udot2_no_clamp:
; GFX906: v_dot2_u32_u16 v{{[0-9]+}}, s{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}{{$}}
; GFX10:  v_dot2_u32_u16 v{{[0-9]+}}, s{{[0-9]+}}, s{{[0-9]+}}, v{{[0-9]+}}{{$}}
define amdgpu_kernel void @test_llvm_amdgcn_udot2_no_clamp(
    i32 addrspace(1)* %r,
    <2 x i16> addrspace(1)* %a,
    <2 x i16> addrspace(1)* %b,
    i32 addrspace(1)* %c) {
entry:
  %a.val = load <2 x i16>, <2 x i16> addrspace(1)* %a
  %b.val = load <2 x i16>, <2 x i16> addrspace(1)* %b
  %c.val = load i32, i32 addrspace(1)* %c
  %r.val = call i32 @llvm.amdgcn.udot2(<2 x i16> %a.val, <2 x i16> %b.val, i32 %c.val, i1 0)
  store i32 %r.val, i32 addrspace(1)* %r
  ret void
}

; GCN-LABEL: {{^}}test_llvm_amdgcn_udot2_op_sel:
; GFX906: v_dot2_u32_u16 v{{[0-9]+}}, 1, v{{[0-9]+}}, s{{[0-9]+}} op_sel:[0,1,0] op_sel_hi:[0,0,1]{{$}}
; GFX10:  v_dot2_u32_u16 v{{[0-9]+}}, 1, v{{[0-9]+}}, s{{[0-9]+}} op_sel:[0,1,0] op_sel_hi:[0,0,1]{{$}}
define amdgpu_kernel void @test_llvm_amdgcn_udot2_op_sel(
    i32 addrspace(1)* %r,
    <2 x i16> addrspace(1)* %b,
    i32 %c) {
entry:
  %id = tail call i32 @llvm.amdgcn.workitem.id.x()
  %b.gep = getelementptr inbounds <2 x i16>, <2 x i16> addrspace(1)* %b, i32 %id
  %b.val = load <2 x i16>, <2 x i16> addrspace(1)* %b.gep
  %b.elt0 = extractelement <2 x i16> %b.val, i32 0
  %b.elt1 = extractelement <2 x i16> %b.val, i32 1
  %b0 = insertelement <2 x i16> undef, i16 %b.elt1, i32 0
  %b1 = insertelement <2 x i16> %b0, i16 %b.elt0, i32 1
  %r.val = call i32 @llvm.amdgcn.udot2(<2 x i16> <i16 1, i16 1>, <2 x i16> %b1, i32 %c, i1 0)
  store i32 %r.val, i32 addrspace(1)* %r
  ret void
}
