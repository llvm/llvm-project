; RUN: llc -march=amdgcn -mcpu=tonga -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GCN,VI %s
; RUN: llc -march=amdgcn -mcpu=gfx900 -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GCN,GFX9 %s

declare float @llvm.amdgcn.ds.fadd.f32(float addrspace(3)* nocapture, float, i32, i32, i1)
declare float @llvm.amdgcn.ds.fmin.f32(float addrspace(3)* nocapture, float, i32, i32, i1)
declare float @llvm.amdgcn.ds.fmax.f32(float addrspace(3)* nocapture, float, i32, i32, i1)

; GCN-LABEL: {{^}}lds_ds_fadd:
; VI-DAG: s_mov_b32 m0
; GFX9-NOT: m0
; GCN-DAG: v_mov_b32_e32 [[V0:v[0-9]+]], 0x42280000
; GCN: ds_add_rtn_f32 [[V2:v[0-9]+]], [[V1:v[0-9]+]], [[V0]] offset:32
; GCN: ds_add_f32 [[V3:v[0-9]+]], [[V0]] offset:64
; GCN: s_waitcnt lgkmcnt(1)
; GCN: ds_add_rtn_f32 {{v[0-9]+}}, {{v[0-9]+}}, [[V2]]
define amdgpu_kernel void @lds_ds_fadd(float addrspace(1)* %out, float addrspace(3)* %ptrf, i32 %idx) {
  %idx.add = add nuw i32 %idx, 4
  %shl0 = shl i32 %idx.add, 3
  %shl1 = shl i32 %idx.add, 4
  %ptr0 = inttoptr i32 %shl0 to float addrspace(3)*
  %ptr1 = inttoptr i32 %shl1 to float addrspace(3)*
  %a1 = call float @llvm.amdgcn.ds.fadd.f32(float addrspace(3)* %ptr0, float 4.2e+1, i32 0, i32 0, i1 false)
  %a2 = call float @llvm.amdgcn.ds.fadd.f32(float addrspace(3)* %ptr1, float 4.2e+1, i32 0, i32 0, i1 false)
  %a3 = call float @llvm.amdgcn.ds.fadd.f32(float addrspace(3)* %ptrf, float %a1, i32 0, i32 0, i1 false)
  store float %a3, float addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}lds_ds_fmin:
; VI-DAG: s_mov_b32 m0
; GFX9-NOT: m0
; GCN-DAG: v_mov_b32_e32 [[V0:v[0-9]+]], 0x42280000
; GCN: ds_min_rtn_f32 [[V2:v[0-9]+]], [[V1:v[0-9]+]], [[V0]] offset:32
; GCN: ds_min_f32 [[V3:v[0-9]+]], [[V0]] offset:64
; GCN: s_waitcnt lgkmcnt(1)
; GCN: ds_min_rtn_f32 {{v[0-9]+}}, {{v[0-9]+}}, [[V2]]
define amdgpu_kernel void @lds_ds_fmin(float addrspace(1)* %out, float addrspace(3)* %ptrf, i32 %idx) {
  %idx.add = add nuw i32 %idx, 4
  %shl0 = shl i32 %idx.add, 3
  %shl1 = shl i32 %idx.add, 4
  %ptr0 = inttoptr i32 %shl0 to float addrspace(3)*
  %ptr1 = inttoptr i32 %shl1 to float addrspace(3)*
  %a1 = call float @llvm.amdgcn.ds.fmin.f32(float addrspace(3)* %ptr0, float 4.2e+1, i32 0, i32 0, i1 false)
  %a2 = call float @llvm.amdgcn.ds.fmin.f32(float addrspace(3)* %ptr1, float 4.2e+1, i32 0, i32 0, i1 false)
  %a3 = call float @llvm.amdgcn.ds.fmin.f32(float addrspace(3)* %ptrf, float %a1, i32 0, i32 0, i1 false)
  store float %a3, float addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}lds_ds_fmax:
; VI-DAG: s_mov_b32 m0
; GFX9-NOT: m0
; GCN-DAG: v_mov_b32_e32 [[V0:v[0-9]+]], 0x42280000
; GCN: ds_max_rtn_f32 [[V2:v[0-9]+]], [[V1:v[0-9]+]], [[V0]] offset:32
; GCN: ds_max_f32 [[V3:v[0-9]+]], [[V0]] offset:64
; GCN: s_waitcnt lgkmcnt(1)
; GCN: ds_max_rtn_f32 {{v[0-9]+}}, {{v[0-9]+}}, [[V2]]
define amdgpu_kernel void @lds_ds_fmax(float addrspace(1)* %out, float addrspace(3)* %ptrf, i32 %idx) {
  %idx.add = add nuw i32 %idx, 4
  %shl0 = shl i32 %idx.add, 3
  %shl1 = shl i32 %idx.add, 4
  %ptr0 = inttoptr i32 %shl0 to float addrspace(3)*
  %ptr1 = inttoptr i32 %shl1 to float addrspace(3)*
  %a1 = call float @llvm.amdgcn.ds.fmax.f32(float addrspace(3)* %ptr0, float 4.2e+1, i32 0, i32 0, i1 false)
  %a2 = call float @llvm.amdgcn.ds.fmax.f32(float addrspace(3)* %ptr1, float 4.2e+1, i32 0, i32 0, i1 false)
  %a3 = call float @llvm.amdgcn.ds.fmax.f32(float addrspace(3)* %ptrf, float %a1, i32 0, i32 0, i1 false)
  store float %a3, float addrspace(1)* %out
  ret void
}
