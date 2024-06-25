; RUN: llc -mtriple=amdgcn -mcpu=gfx900 -verify-machineinstrs < %s | FileCheck %s -check-prefix=GCN
; RUN: llc -mtriple=amdgcn -mcpu=gfx1010 -verify-machineinstrs < %s | FileCheck %s -check-prefix=GCN
; RUN: llc -mtriple=amdgcn -mcpu=gfx1100 -verify-machineinstrs < %s | FileCheck %s -check-prefix=GCN
; RUN: llc -mtriple=amdgcn -mcpu=gfx1150 -verify-machineinstrs < %s | FileCheck %s -check-prefix=GCN

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

; Fails to combine because v_mul_lo_u32 has no e32 or dpp form.
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

declare i32 @llvm.amdgcn.workitem.id.x()
declare i32 @llvm.amdgcn.update.dpp.i32(i32, i32, i32, i32, i32, i1) #0
declare float @llvm.ceil.f32(float)

attributes #0 = { nounwind readnone convergent }
