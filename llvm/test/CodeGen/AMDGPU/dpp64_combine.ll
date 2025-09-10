; RUN: llc -mtriple=amdgcn -mcpu=gfx90a < %s | FileCheck %s -check-prefixes=GCN,DPP64,GFX90A,DPP64-GFX9 -DCTL=row_newbcast
; RUN: llc -mtriple=amdgcn -mcpu=gfx942 < %s | FileCheck %s -check-prefixes=GCN,DPP64,DPPMOV64,DPP64-GFX9,GFX942 -DCTL=row_newbcast
; RUN: llc -mtriple=amdgcn -mcpu=gfx1010 < %s | FileCheck %s -check-prefixes=GCN,DPP32,GFX10PLUS,GFX10 -DCTL=row_share
; RUN: llc -mtriple=amdgcn -mcpu=gfx1100 < %s | FileCheck %s -check-prefixes=GCN,DPP32,GFX10PLUS,GFX11 -DCTL=row_share
; RUN: llc -mtriple=amdgcn -mcpu=gfx1250 < %s | FileCheck %s -check-prefixes=GCN,DPP32,GFX1250 -DCTL=row_share

; GCN-LABEL: {{^}}dpp64_ceil:
; GCN:           global_load_{{dwordx2|b64}} [[V:v\[[0-9:]+\]]],
; DPP64:         v_ceil_f64_dpp [[V]], [[V]] [[CTL]]:1 row_mask:0xf bank_mask:0xf bound_ctrl:1{{$}}
; DPP32-COUNT-2: v_mov_b32_dpp v{{[0-9]+}}, v{{[0-9]+}} [[CTL]]:1 row_mask:0xf bank_mask:0xf bound_ctrl:1{{$}}
define amdgpu_kernel void @dpp64_ceil(ptr addrspace(1) %arg, i64 %in1) {
  %id = tail call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr inbounds i64, ptr addrspace(1) %arg, i32 %id
  %load = load i64, ptr addrspace(1) %gep
  %tmp0 = call i64 @llvm.amdgcn.update.dpp.i64(i64 %in1, i64 %load, i32 337, i32 15, i32 15, i1 1) #0
  %tmp1 = bitcast i64 %tmp0 to double
  %round = tail call double @llvm.ceil.f64(double %tmp1)
  %tmp2 = bitcast double %round to i64
  store i64 %tmp2, ptr addrspace(1) %gep
  ret void
}

; GCN-LABEL: {{^}}dpp64_rcp:
; GCN:           global_load_{{dwordx2|b64}} [[V:v\[[0-9:]+\]]],
; DPP64-GFX9:    v_rcp_f64_dpp [[V]], [[V]] [[CTL]]:1 row_mask:0xf bank_mask:0xf bound_ctrl:1{{$}}
; DPP32-COUNT-2: v_mov_b32_dpp v{{[0-9]+}}, v{{[0-9]+}} [[CTL]]:1 row_mask:0xf bank_mask:0xf bound_ctrl:1{{$}}
define amdgpu_kernel void @dpp64_rcp(ptr addrspace(1) %arg, i64 %in1) {
  %id = tail call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr inbounds i64, ptr addrspace(1) %arg, i32 %id
  %load = load i64, ptr addrspace(1) %gep
  %tmp0 = call i64 @llvm.amdgcn.update.dpp.i64(i64 %in1, i64 %load, i32 337, i32 15, i32 15, i1 1) #0
  %tmp1 = bitcast i64 %tmp0 to double
  %rcp = call double @llvm.amdgcn.rcp.f64(double %tmp1)
  %tmp2 = bitcast double %rcp to i64
  store i64 %tmp2, ptr addrspace(1) %gep
  ret void
}

; GCN-LABEL: {{^}}dpp64_rcp_unsupported_ctl:
; GCN-COUNT-2: v_mov_b32_dpp v{{[0-9]+}}, v{{[0-9]+}} quad_perm:[1,0,0,0] row_mask:0xf bank_mask:0xf bound_ctrl:1{{$}}
; GCN:         v_rcp_f64_e32
define amdgpu_kernel void @dpp64_rcp_unsupported_ctl(ptr addrspace(1) %arg, i64 %in1) {
  %id = tail call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr inbounds i64, ptr addrspace(1) %arg, i32 %id
  %load = load i64, ptr addrspace(1) %gep
  %tmp0 = call i64 @llvm.amdgcn.update.dpp.i64(i64 %in1, i64 %load, i32 1, i32 15, i32 15, i1 1) #0
  %tmp1 = bitcast i64 %tmp0 to double
  %rcp = fdiv fast double 1.0, %tmp1
  %tmp2 = bitcast double %rcp to i64
  store i64 %tmp2, ptr addrspace(1) %gep
  ret void
}

; GCN-LABEL: {{^}}dpp64_div:
; GCN:               global_load_{{dwordx2|b64}} [[V:v\[[0-9:]+\]]],
; DPPMOV64:          v_mov_b64_dpp v[{{[0-9:]+}}], [[V]] [[CTL]]:1 row_mask:0xf bank_mask:0xf bound_ctrl:1{{$}}
; GFX90A-COUNT-2:    v_mov_b32_dpp v{{[0-9]+}}, v{{[0-9]+}} [[CTL]]:1 row_mask:0xf bank_mask:0xf bound_ctrl:1{{$}}
; DPP32-COUNT-2:     v_mov_b32_dpp v{{[0-9]+}}, v{{[0-9]+}} [[CTL]]:1 row_mask:0xf bank_mask:0xf bound_ctrl:1{{$}}
; GCN:               v_div_scale_f64
; GCN:               v_rcp_f64_e32
define amdgpu_kernel void @dpp64_div(ptr addrspace(1) %arg, i64 %in1) {
  %id = tail call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr inbounds i64, ptr addrspace(1) %arg, i32 %id
  %load = load i64, ptr addrspace(1) %gep
  %tmp0 = call i64 @llvm.amdgcn.update.dpp.i64(i64 %in1, i64 %load, i32 337, i32 15, i32 15, i1 1) #0
  %tmp1 = bitcast i64 %tmp0 to double
  %rcp = fdiv double 15.0, %tmp1
  %tmp2 = bitcast double %rcp to i64
  store i64 %tmp2, ptr addrspace(1) %gep
  ret void
}

; On GFX9 it fails to combine because v_mul_lo_u32 has no e32 or dpp form.
; GCN-LABEL: {{^}}dpp_mul_row_share:
; GCN: global_load_{{dword|b32}} [[V:v[0-9]+]],
; DPP64-GFX9: v_mov_b32_e32 [[V2:v[0-9]+]], [[V]]
; DPP64-GFX9: v_mov_b32_dpp [[V2]], [[V2]] {{row_share|row_newbcast}}:0 row_mask:0xf bank_mask:0xf bound_ctrl:1{{$}}
; DPP64-GFX9: v_mul_lo_u32 [[V]], [[V2]], [[V]]{{$}}
; GFX1250: v_mov_b32_e32 [[V2:v[0-9]+]], [[V]]
; GFX1250: v_mov_b32_dpp [[V2]], [[V2]] {{row_share|row_newbcast}}:0 row_mask:0xf bank_mask:0xf bound_ctrl:1{{$}}
; GFX1250: v_mul_lo_u32 [[V]], [[V2]], [[V]]{{$}}
define amdgpu_kernel void @dpp_mul_row_share(ptr addrspace(1) %arg) {
  %id = tail call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr inbounds i32, ptr addrspace(1) %arg, i32 %id
  %load = load i32, ptr addrspace(1) %gep
  %tmp0 = call i32 @llvm.amdgcn.update.dpp.i32(i32 %load, i32 %load, i32 336, i32 15, i32 15, i1 1)
  %mul = mul i32 %tmp0, %load
  store i32 %mul, ptr addrspace(1) %gep
  ret void
}

; GCN-LABEL: {{^}}dpp64_loop:
; GCN: v_mov_b32_dpp
; DPP64: v_mov_b32_dpp
; GFX90A: v_add_co_u32_e32
; GFX90A: v_addc_co_u32_e32
; GFX942: v_lshl_add_u64
; GFX10PLUS: v_mov_b32_dpp
; GFX10PLUS: v_add_co_u32
; GFX10: v_add_co_ci_u32_e32
; GFX11: v_add_co_ci_u32_e64
define amdgpu_cs void @dpp64_loop(i64 %arg, i64 %val) {
bb:
  br label %bb1
bb1:
  %i = call i64 @llvm.amdgcn.update.dpp.i64(i64 poison, i64 %val, i32 0, i32 0, i32 0, i1 false)
  %i2 = add i64 %i, %arg
  %i3 = atomicrmw add ptr addrspace(1) null, i64 %i2 monotonic, align 8
  br label %bb1
}

declare i32 @llvm.amdgcn.workitem.id.x()
declare i64 @llvm.amdgcn.update.dpp.i64(i64, i64, i32, i32, i32, i1) #0
declare double @llvm.ceil.f64(double)
declare double @llvm.amdgcn.rcp.f64(double)

attributes #0 = { nounwind readnone convergent }
