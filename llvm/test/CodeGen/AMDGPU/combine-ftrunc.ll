; RUN: llc -march=amdgcn -mcpu=tonga -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s

; GCN-LABEL: {{^}}combine_ftrunc_frint_f64:
; GCN: v_rndne_f64_e32 [[RND:v\[[0-9:]+\]]],
; GCN: flat_store_dwordx2 v[{{[0-9:]+}}], [[RND]]
define amdgpu_kernel void @combine_ftrunc_frint_f64(double addrspace(1)* %p) {
  %v = load double, double addrspace(1)* %p, align 8
  %round = tail call double @llvm.rint.f64(double %v)
  %trunc = tail call double @llvm.trunc.f64(double %round)
  store double %trunc, double addrspace(1)* %p, align 8
  ret void
}

; GCN-LABEL: {{^}}combine_ftrunc_frint_f32:
; GCN: v_rndne_f32_e32 [[RND:v[0-9]+]],
; GCN: flat_store_dword v[{{[0-9:]+}}], [[RND]]
define amdgpu_kernel void @combine_ftrunc_frint_f32(float addrspace(1)* %p) {
  %v = load float, float addrspace(1)* %p, align 4
  %round = tail call float @llvm.rint.f32(float %v)
  %trunc = tail call float @llvm.trunc.f32(float %round)
  store float %trunc, float addrspace(1)* %p, align 4
  ret void
}

; GCN-LABEL: {{^}}combine_ftrunc_frint_v2f32:
; GCN: s_load_dwordx2
; GCN: s_load_dwordx2 s{{\[}}[[SRC1:[0-9]+]]:[[SRC2:[0-9]+]]{{\]}}
; GCN-DAG: v_rndne_f32_e32 v[[RND1:[0-9]+]], s[[SRC1]]
; GCN-DAG: v_rndne_f32_e32 v[[RND2:[0-9]+]], s[[SRC2]]
; GCN: flat_store_dwordx2 v[{{[0-9:]+}}], v{{\[}}[[RND1]]:[[RND2]]{{\]}}
define amdgpu_kernel void @combine_ftrunc_frint_v2f32(<2 x float> addrspace(1)* %p) {
  %v = load <2 x float>, <2 x float> addrspace(1)* %p, align 8
  %round = tail call <2 x float> @llvm.rint.v2f32(<2 x float> %v)
  %trunc = tail call <2 x float> @llvm.trunc.v2f32(<2 x float> %round)
  store <2 x float> %trunc, <2 x float> addrspace(1)* %p, align 8
  ret void
}

; GCN-LABEL: {{^}}combine_ftrunc_fceil_f32:
; GCN: v_ceil_f32_e32 [[RND:v[0-9]+]],
; GCN: flat_store_dword v[{{[0-9:]+}}], [[RND]]
define amdgpu_kernel void @combine_ftrunc_fceil_f32(float addrspace(1)* %p) {
  %v = load float, float addrspace(1)* %p, align 4
  %round = tail call float @llvm.ceil.f32(float %v)
  %trunc = tail call float @llvm.trunc.f32(float %round)
  store float %trunc, float addrspace(1)* %p, align 4
  ret void
}

; GCN-LABEL: {{^}}combine_ftrunc_ffloor_f32:
; GCN: v_floor_f32_e32 [[RND:v[0-9]+]],
; GCN: flat_store_dword v[{{[0-9:]+}}], [[RND]]
define amdgpu_kernel void @combine_ftrunc_ffloor_f32(float addrspace(1)* %p) {
  %v = load float, float addrspace(1)* %p, align 4
  %round = tail call float @llvm.floor.f32(float %v)
  %trunc = tail call float @llvm.trunc.f32(float %round)
  store float %trunc, float addrspace(1)* %p, align 4
  ret void
}

; GCN-LABEL: {{^}}combine_ftrunc_fnearbyint_f32:
; GCN: v_rndne_f32_e32 [[RND:v[0-9]+]],
; GCN: flat_store_dword v[{{[0-9:]+}}], [[RND]]
define amdgpu_kernel void @combine_ftrunc_fnearbyint_f32(float addrspace(1)* %p) {
  %v = load float, float addrspace(1)* %p, align 4
  %round = tail call float @llvm.nearbyint.f32(float %v)
  %trunc = tail call float @llvm.trunc.f32(float %round)
  store float %trunc, float addrspace(1)* %p, align 4
  ret void
}

; GCN-LABEL: {{^}}combine_ftrunc_ftrunc_f32:
; GCN: s_load_dword [[SRC:s[0-9]+]],
; GCN: v_trunc_f32_e32 [[RND:v[0-9]+]], [[SRC]]
; GCN: flat_store_dword v[{{[0-9:]+}}], [[RND]]
define amdgpu_kernel void @combine_ftrunc_ftrunc_f32(float addrspace(1)* %p) {
  %v = load float, float addrspace(1)* %p, align 4
  %round = tail call float @llvm.trunc.f32(float %v)
  %trunc = tail call float @llvm.trunc.f32(float %round)
  store float %trunc, float addrspace(1)* %p, align 4
  ret void
}

declare double @llvm.trunc.f64(double)
declare float @llvm.trunc.f32(float)
declare <2 x float> @llvm.trunc.v2f32(<2 x float>)
declare double @llvm.rint.f64(double)
declare float @llvm.rint.f32(float)
declare <2 x float> @llvm.rint.v2f32(<2 x float>)
declare float @llvm.ceil.f32(float)
declare float @llvm.floor.f32(float)
declare float @llvm.nearbyint.f32(float)
