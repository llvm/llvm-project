; RUN: llc -march=amdgcn -mcpu=gfx900 -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,GFX9 %s
; RUN: llc -march=amdgcn -mcpu=gfx1010 -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,GFX10 %s
; RUN: llc -march=amdgcn -mcpu=gfx1100 -verify-machineinstrs -amdgpu-enable-vopd=0 < %s | FileCheck -check-prefixes=GCN,GFX10 %s

; GCN-LABEL: {{^}}test_add_lit:
; GFX10: v_add_co_u32{{(_e64)?}} v{{[0-9]+}}, vcc_lo, 0x80992bff, v{{[0-9]+}}
; GFX10: v_add_co_ci_u32_e32 v{{[0-9]+}}, vcc_lo, 0xe7, v{{[0-9]+}}, vcc_lo
; GFX9:  v_mov_b32_e32 [[C2:v[0-9]+]], 0xe7
; GFX9:  v_add_co_u32_e32 v{{[0-9]+}}, vcc, 0x80992bff, v{{[0-9]+}}
; GFX9:  v_addc_co_u32_e32 v{{[0-9]+}}, vcc, v{{[0-9]+}}, [[C2]], vcc
define amdgpu_kernel void @test_add_lit(i64 addrspace(1)* %p) {
  %id = tail call i32 @llvm.amdgcn.workitem.id.x()
  %ptr = getelementptr inbounds i64, i64 addrspace(1)* %p, i32 %id
  %load = load i64, i64 addrspace(1)* %ptr, align 8
  %add = add nsw i64 %load, 994294967295
  store i64 %add, i64 addrspace(1)* %ptr, align 8
  ret void
}

; GCN-LABEL: {{^}}test_cndmask_lit:
; GFX10: v_cndmask_b32_e32 v{{[0-9]+}}, 0x3039, v{{[0-9]+}}, vcc_lo
; GFX9:  v_mov_b32_e32 [[C:v[0-9]+]], 0x3039
; GFX9:  v_cndmask_b32_e32 v{{[0-9]+}}, [[C]], v{{[0-9]+}}, vcc
define amdgpu_kernel void @test_cndmask_lit(i32 addrspace(1)* %p) {
  %id = tail call i32 @llvm.amdgcn.workitem.id.x()
  %n = add nuw nsw i32 %id, 1
  %p1 = getelementptr inbounds i32, i32 addrspace(1)* %p, i32 %id
  %v1 = load i32, i32 addrspace(1)* %p1, align 4
  %p2 = getelementptr inbounds i32, i32 addrspace(1)* %p, i32 %n
  %v2 = load i32, i32 addrspace(1)* %p2, align 4
  %cmp = icmp sgt i32 %v1, 0
  %sel = select i1 %cmp, i32 12345, i32 %v2
  store i32 %sel, i32 addrspace(1)* %p1, align 4
  ret void
}

; GCN-LABEL: {{^}}test_bfe_2lit_s:
; GFX10: v_mov_b32_e32 [[C1:v[0-9]+]], 0xddd5
; GFX10: v_bfe_u32 v{{[0-9]+}}, 0x3039, s{{[0-9]+}}, [[C1]]
; GFX9-DAG: v_mov_b32_e32 [[C2:v[0-9]+]], 0xddd5
; GFX9-DAG: s_movk_i32 [[C1:s[0-9]+]], 0x3039
; GFX9:     v_bfe_u32 v{{[0-9]+}}, [[C1]], v{{[0-9]+}}, [[C2]]
define amdgpu_kernel void @test_bfe_2lit_s(i32 addrspace(1)* %p, i32 %src) {
  %bfe = call i32 @llvm.amdgcn.ubfe.i32(i32 12345, i32 %src, i32 56789)
  store i32 %bfe, i32 addrspace(1)* %p, align 4
  ret void
}

; GCN-LABEL: {{^}}test_bfe_2lit_v:
; GFX10: s_movk_i32 [[C1:s[0-9]+]], 0x3039
; GFX10: v_bfe_u32 v{{[0-9]+}}, [[C1]], v{{[0-9]+}}, 0xddd5
; GFX9-DAG: v_mov_b32_e32 [[C2:v[0-9]+]], 0xddd5
; GFX9-DAG: s_movk_i32 [[C1:s[0-9]+]], 0x3039
; GFX9:     v_bfe_u32 v{{[0-9]+}}, [[C1]], v{{[0-9]+}}, [[C2]]
define amdgpu_kernel void @test_bfe_2lit_v(i32 addrspace(1)* %p) {
  %id = tail call i32 @llvm.amdgcn.workitem.id.x()
  %ptr = getelementptr inbounds i32, i32 addrspace(1)* %p, i32 %id
  %load = load i32, i32 addrspace(1)* %ptr, align 4
  %bfe = call i32 @llvm.amdgcn.ubfe.i32(i32 12345, i32 %load, i32 56789)
  store i32 %bfe, i32 addrspace(1)* %ptr, align 4
  ret void
}

declare i32 @llvm.amdgcn.workitem.id.x()
declare i32 @llvm.amdgcn.ubfe.i32(i32, i32, i32)
