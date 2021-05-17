; RUN: llc -march=amdgcn -mcpu=gfx90a -verify-machineinstrs < %s | FileCheck %s -check-prefixes=GCN,DPP64,GFX90A
; RUN: llc -march=amdgcn -mcpu=gfx1010 -verify-machineinstrs < %s | FileCheck %s -check-prefixes=GCN,DPP32,GFX10

; GCN-LABEL: {{^}}dpp64_ceil:
; GCN:           global_load_dwordx2 [[V:v\[[0-9:]+\]]],
; DPP64:         v_ceil_f64_dpp [[V]], [[V]] row_newbcast:1 row_mask:0xf bank_mask:0xf bound_ctrl:1{{$}}
; DPP32-COUNT-2: v_mov_b32_dpp v{{[0-9]+}}, v{{[0-9]+}} row_share:1 row_mask:0xf bank_mask:0xf bound_ctrl:1{{$}}
define amdgpu_kernel void @dpp64_ceil(i64 addrspace(1)* %arg, i64 %in1) {
  %id = tail call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr inbounds i64, i64 addrspace(1)* %arg, i32 %id
  %load = load i64, i64 addrspace(1)* %gep
  %tmp0 = call i64 @llvm.amdgcn.update.dpp.i64(i64 %in1, i64 %load, i32 337, i32 15, i32 15, i1 1) #0
  %tmp1 = bitcast i64 %tmp0 to double
  %round = tail call double @llvm.ceil.f64(double %tmp1)
  %tmp2 = bitcast double %round to i64
  store i64 %tmp2, i64 addrspace(1)* %gep
  ret void
}

; GCN-LABEL: {{^}}dpp64_rcp:
; GCN:           global_load_dwordx2 [[V:v\[[0-9:]+\]]],
; DPP64:         v_rcp_f64_dpp [[V]], [[V]] row_newbcast:1 row_mask:0xf bank_mask:0xf bound_ctrl:1{{$}}
; DPP32-COUNT-2: v_mov_b32_dpp v{{[0-9]+}}, v{{[0-9]+}} row_share:1 row_mask:0xf bank_mask:0xf bound_ctrl:1{{$}}
define amdgpu_kernel void @dpp64_rcp(i64 addrspace(1)* %arg, i64 %in1) {
  %id = tail call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr inbounds i64, i64 addrspace(1)* %arg, i32 %id
  %load = load i64, i64 addrspace(1)* %gep
  %tmp0 = call i64 @llvm.amdgcn.update.dpp.i64(i64 %in1, i64 %load, i32 337, i32 15, i32 15, i1 1) #0
  %tmp1 = bitcast i64 %tmp0 to double
  %rcp = call double @llvm.amdgcn.rcp.f64(double %tmp1)
  %tmp2 = bitcast double %rcp to i64
  store i64 %tmp2, i64 addrspace(1)* %gep
  ret void
}

; GCN-LABEL: {{^}}dpp64_rcp_unsupported_ctl:
; GCN-COUNT-2: v_mov_b32_dpp v{{[0-9]+}}, v{{[0-9]+}} quad_perm:[1,0,0,0] row_mask:0xf bank_mask:0xf bound_ctrl:1{{$}}
; GCN:         v_rcp_f64_e32
define amdgpu_kernel void @dpp64_rcp_unsupported_ctl(i64 addrspace(1)* %arg, i64 %in1) {
  %id = tail call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr inbounds i64, i64 addrspace(1)* %arg, i32 %id
  %load = load i64, i64 addrspace(1)* %gep
  %tmp0 = call i64 @llvm.amdgcn.update.dpp.i64(i64 %in1, i64 %load, i32 1, i32 15, i32 15, i1 1) #0
  %tmp1 = bitcast i64 %tmp0 to double
  %rcp = fdiv fast double 1.0, %tmp1
  %tmp2 = bitcast double %rcp to i64
  store i64 %tmp2, i64 addrspace(1)* %gep
  ret void
}

; GCN-LABEL: {{^}}dpp64_div:
; GCN:            global_load_dwordx2 [[V:v\[[0-9:]+\]]],
; GFX90A-COUNT-2: v_mov_b32_dpp v{{[0-9]+}}, v{{[0-9]+}} row_newbcast:1 row_mask:0xf bank_mask:0xf bound_ctrl:1{{$}}
; GFX10-COUNT-2:  v_mov_b32_dpp v{{[0-9]+}}, v{{[0-9]+}} row_share:1 row_mask:0xf bank_mask:0xf bound_ctrl:1{{$}}
; GCN:            v_div_scale_f64
; GCN:            v_rcp_f64_e32
define amdgpu_kernel void @dpp64_div(i64 addrspace(1)* %arg, i64 %in1) {
  %id = tail call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr inbounds i64, i64 addrspace(1)* %arg, i32 %id
  %load = load i64, i64 addrspace(1)* %gep
  %tmp0 = call i64 @llvm.amdgcn.update.dpp.i64(i64 %in1, i64 %load, i32 337, i32 15, i32 15, i1 1) #0
  %tmp1 = bitcast i64 %tmp0 to double
  %rcp = fdiv double 15.0, %tmp1
  %tmp2 = bitcast double %rcp to i64
  store i64 %tmp2, i64 addrspace(1)* %gep
  ret void
}

declare i32 @llvm.amdgcn.workitem.id.x()
declare i64 @llvm.amdgcn.update.dpp.i64(i64, i64, i32, i32, i32, i1) #0
declare double @llvm.ceil.f64(double)
declare double @llvm.amdgcn.rcp.f64(double)

attributes #0 = { nounwind readnone convergent }
