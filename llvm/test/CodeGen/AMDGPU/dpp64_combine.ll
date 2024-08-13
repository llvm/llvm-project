; RUN: llc -mtriple=amdgcn -mcpu=gfx90a -verify-machineinstrs < %s | FileCheck %s -check-prefixes=GCN,DPP64,GFX90A
; RUN: llc -mtriple=amdgcn -mcpu=gfx940 -verify-machineinstrs < %s | FileCheck %s -check-prefixes=GCN,DPP64,DPPMOV64,GFX940
; RUN: llc -mtriple=amdgcn -mcpu=gfx1010 -verify-machineinstrs < %s | FileCheck %s -check-prefixes=GCN,DPP32,GFX10PLUS,GFX10
; RUN: llc -mtriple=amdgcn -mcpu=gfx1100 -verify-machineinstrs < %s | FileCheck %s -check-prefixes=GCN,DPP32,GFX10PLUS,GFX11

; GCN-LABEL: {{^}}dpp64_ceil:
; GCN:           global_load_{{dwordx2|b64}} [[V:v\[[0-9:]+\]]],
; DPP64:         v_ceil_f64_dpp [[V]], [[V]] row_newbcast:1 row_mask:0xf bank_mask:0xf bound_ctrl:1{{$}}
; DPP32-COUNT-2: v_mov_b32_dpp v{{[0-9]+}}, v{{[0-9]+}} row_share:1 row_mask:0xf bank_mask:0xf bound_ctrl:1{{$}}
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
; DPP64:         v_rcp_f64_dpp [[V]], [[V]] row_newbcast:1 row_mask:0xf bank_mask:0xf bound_ctrl:1{{$}}
; DPP32-COUNT-2: v_mov_b32_dpp v{{[0-9]+}}, v{{[0-9]+}} row_share:1 row_mask:0xf bank_mask:0xf bound_ctrl:1{{$}}
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
; DPPMOV64:          v_mov_b64_dpp v[{{[0-9:]+}}], [[V]] row_newbcast:1 row_mask:0xf bank_mask:0xf bound_ctrl:1{{$}}
; GFX90A-COUNT-2:    v_mov_b32_dpp v{{[0-9]+}}, v{{[0-9]+}} row_newbcast:1 row_mask:0xf bank_mask:0xf bound_ctrl:1{{$}}
; GFX10PLUS-COUNT-2: v_mov_b32_dpp v{{[0-9]+}}, v{{[0-9]+}} row_share:1 row_mask:0xf bank_mask:0xf bound_ctrl:1{{$}}
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

; GCN-LABEL: {{^}}dpp64_loop:
; GCN: v_mov_b32_dpp
; DPP64: v_mov_b32_dpp
; GFX90A: v_add_co_u32_e32
; GFX90A: v_addc_co_u32_e32
; GFX940: v_lshl_add_u64
; GFX10: v_mov_b32_dpp
; GFX10: v_add_co_u32
; GFX10: v_add_co_ci_u32_e32
; GFX11: v_add_co_u32_e64_dpp
; GFX11: v_add_co_ci_u32_e32
define amdgpu_cs void @dpp64_loop(i64 %arg) {
bb:
  br label %bb1
bb1:
  %i = call i64 @llvm.amdgcn.update.dpp.i64(i64 0, i64 0, i32 0, i32 0, i32 0, i1 false)
  %i2 = add i64 %i, %arg
  %i3 = atomicrmw add ptr addrspace(1) null, i64 %i2 monotonic, align 8
  br label %bb1
}

declare i32 @llvm.amdgcn.workitem.id.x()
declare i64 @llvm.amdgcn.update.dpp.i64(i64, i64, i32, i32, i32, i1) #0
declare double @llvm.ceil.f64(double)
declare double @llvm.amdgcn.rcp.f64(double)

attributes #0 = { nounwind readnone convergent }
