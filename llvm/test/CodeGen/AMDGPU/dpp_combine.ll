; RUN: llc -mtriple=amdgcn -mcpu=gfx900 < %s | FileCheck %s -check-prefixes=GCN,GFX9GFX10
; RUN: llc -mtriple=amdgcn -mcpu=gfx1010 < %s | FileCheck %s -check-prefixes=GCN,GFX9GFX10
; RUN: llc -mtriple=amdgcn -mcpu=gfx1100 -mattr=+real-true16 < %s | FileCheck %s -check-prefixes=GCN,GFX11-TRUE16
; RUN: llc -mtriple=amdgcn -mcpu=gfx1100 -mattr=-real-true16 < %s | FileCheck %s -check-prefixes=GCN,GFX11-FAKE16,GFX11
; RUN: llc -mtriple=amdgcn -mcpu=gfx1150 -mattr=+real-true16 < %s | FileCheck %s -check-prefixes=GCN,GFX11-TRUE16
; RUN: llc -mtriple=amdgcn -mcpu=gfx1150 -mattr=-real-true16 < %s | FileCheck %s -check-prefixes=GCN,GFX11-FAKE16
; RUN: llc -mtriple=amdgcn -mcpu=gfx1200 -mattr=+real-true16 < %s | FileCheck %s -check-prefixes=GCN,GFX11-TRUE16
; RUN: llc -mtriple=amdgcn -mcpu=gfx1200 -mattr=-real-true16 < %s | FileCheck %s -check-prefixes=GCN,GFX11-FAKE16,GFX12
; RUN: llc -mtriple=amdgcn -mcpu=gfx1251 -mattr=+real-true16 < %s | FileCheck %s -check-prefixes=GCN,GFX11-TRUE16
; RUN: llc -mtriple=amdgcn -mcpu=gfx1251 -mattr=-real-true16 < %s | FileCheck %s -check-prefixes=GCN,GFX11-FAKE16

; GCN-LABEL: {{^}}dpp_add:
; GCN: global_load_{{dword|b32}} [[V:v[0-9]+]],
; GCN: v_add_{{(nc_)?}}u32_dpp [[V]], [[V]], [[V]] quad_perm:[1,0,0,0] row_mask:0xf bank_mask:0xf bound_ctrl:1{{$}}
define amdgpu_kernel void @dpp_add(ptr addrspace(1) %arg) {
  %id = tail call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr inbounds i32, ptr addrspace(1) %arg, i32 %id
  %load = load i32, ptr addrspace(1) %gep
  %tmp0 = call i32 @llvm.amdgcn.update.dpp.i32(i32 %load, i32 %load, i32 1, i32 15, i32 15, i1 1) #0
  %add = add i32 %tmp0, %load
  store i32 %add, ptr addrspace(1) %gep
  ret void
}

; GCN-LABEL: {{^}}dpp_ceil:
; GCN: global_load_{{dword|b32}} [[V:v[0-9]+]],
; GCN: v_ceil_f32_dpp [[V]], [[V]] quad_perm:[1,0,0,0] row_mask:0xf bank_mask:0xf bound_ctrl:1{{$}}
define amdgpu_kernel void @dpp_ceil(ptr addrspace(1) %arg) {
  %id = tail call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr inbounds i32, ptr addrspace(1) %arg, i32 %id
  %load = load i32, ptr addrspace(1) %gep
  %tmp0 = call i32 @llvm.amdgcn.update.dpp.i32(i32 %load, i32 %load, i32 1, i32 15, i32 15, i1 1) #0
  %tmp1 = bitcast i32 %tmp0 to float
  %round = tail call float @llvm.ceil.f32(float %tmp1)
  %tmp2 = bitcast float %round to i32
  store i32 %tmp2, ptr addrspace(1) %gep
  ret void
}

; GCN-LABEL: {{^}}dpp_fadd:
; GCN: global_load_{{dword|b32}} [[V:v[0-9]+]],
; GCN: v_add_f32_dpp [[V]], [[V]], [[V]] quad_perm:[1,0,0,0] row_mask:0xf bank_mask:0xf bound_ctrl:1{{$}}
define amdgpu_kernel void @dpp_fadd(ptr addrspace(1) %arg) {
  %id = tail call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr inbounds i32, ptr addrspace(1) %arg, i32 %id
  %load = load i32, ptr addrspace(1) %gep
  %tmp0 = call i32 @llvm.amdgcn.update.dpp.i32(i32 %load, i32 %load, i32 1, i32 15, i32 15, i1 1) #0
  %tmp1 = bitcast i32 %tmp0 to float
  %t = bitcast i32 %load to float
  %add = fadd float %tmp1, %t
  %tmp2 = bitcast float %add to i32
  store i32 %tmp2, ptr addrspace(1) %gep
  ret void
}

; Fails to combine prior to gfx1251 because v_mul_lo_u32 has no e32 or dpp form.
; Fails to combine on gfx1251 because DPP control value is invalid for DP DPP and v_mul_lo_u32 is
; classified as DP DPP.
; GCN-LABEL: {{^}}dpp_mul:
; GCN: global_load_{{dword|b32}} [[V:v[0-9]+]],
; GCN: v_mov_b32_e32 [[V2:v[0-9]+]], [[V]]
; GCN: v_mov_b32_dpp [[V2]], [[V2]] quad_perm:[1,0,0,0] row_mask:0xf bank_mask:0xf bound_ctrl:1{{$}}
; GCN: v_mul_lo_u32 [[V]], [[V2]], [[V]]{{$}}
define amdgpu_kernel void @dpp_mul(ptr addrspace(1) %arg) {
  %id = tail call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr inbounds i32, ptr addrspace(1) %arg, i32 %id
  %load = load i32, ptr addrspace(1) %gep
  %tmp0 = call i32 @llvm.amdgcn.update.dpp.i32(i32 %load, i32 %load, i32 1, i32 15, i32 15, i1 1)
  %mul = mul i32 %tmp0, %load
  store i32 %mul, ptr addrspace(1) %gep
  ret void
}

; It is not expected to see a sequence of v_mov_b32_dpp feeding into a 16 bit instruction
; GCN-LABEL: {{^}}dpp_fadd_f16:
; GFX9GFX10: global_load_{{dword|b32}} [[V:v[0-9]+]],
; GFX9GFX10: v_add_f16_dpp [[V]], [[V]], [[V]] quad_perm:[1,0,0,0] row_mask:0xf bank_mask:0xf bound_ctrl:1{{$}}
; GFX11-TRUE16: v_mov_b32_dpp {{v[0-9]+}}, {{v[0-9]+}} quad_perm:[1,0,0,0] row_mask:0xf bank_mask:0xf bound_ctrl:1
; GFX11-TRUE16: v_add_f16_e32
; GFX11-FAKE16: global_load_{{dword|b32}} [[V:v[0-9]+]],
; GFX11-FAKE16: v_add_f16_e64_dpp [[V]], [[V]], [[V]] quad_perm:[1,0,0,0] row_mask:0xf bank_mask:0xf bound_ctrl:1
define amdgpu_kernel void @dpp_fadd_f16(ptr addrspace(1) %arg) {
  %id = tail call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr inbounds i32, ptr addrspace(1) %arg, i32 %id
  %load = load i32, ptr addrspace(1) %gep
  %tmp0 = call i32 @llvm.amdgcn.update.dpp.i32(i32 %load, i32 %load, i32 1, i32 15, i32 15, i1 1) #0
  %tmp01 = trunc i32 %tmp0 to i16
  %tmp1 = bitcast i16 %tmp01 to half
  %tt = trunc i32 %load to i16
  %t = bitcast i16 %tt to half
  %add = fadd half %tmp1, %t
  %tmp2 = bitcast half %add to i16
  %tmp3 = zext i16 %tmp2 to i32
  store i32 %tmp3, ptr addrspace(1) %gep
  ret void
}

; GCN-LABEL: {{^}}dpp_src1_sgpr:
; GFX11: v_add_nc_u16 {{v[0-9]+}}, {{s[0-9]+}}, {{v[0-9]+}}
; GFX12: v_add_nc_u16_e64_dpp {{v[0-9]+}}, {{v[0-9]+}}, {{s[0-9]+}}
define amdgpu_kernel void @dpp_src1_sgpr(ptr addrspace(1) %out, i32 %in) {
  %5 = trunc i32 %in to i8
  %6 = shl i8 %5, 3
  %7 = sext i8 %6 to i32
  %8 = tail call i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %7, i32 280, i32 15, i32 15, i1 true)
  %9 = trunc i32 %8 to i8
  %10 = add i8 %6, %9
  %11 = sext i8 %10 to i32
  store i32 %11, ptr addrspace(1) %out
  ret void
}

declare i32 @llvm.amdgcn.workitem.id.x()
declare i32 @llvm.amdgcn.update.dpp.i32(i32, i32, i32, i32, i32, i1) #0
declare float @llvm.ceil.f32(float)

attributes #0 = { nounwind readnone convergent }
