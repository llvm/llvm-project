; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=tahiti -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GCN,CI,CIT %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=hawaii -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GCN,CI,CIH %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GCN,GFX9 %s

; GCN-LABEL: {{^}}is_local_vgpr:
; GCN-DAG: {{flat|global|buffer}}_load_dwordx2 v{{\[[0-9]+}}:[[PTR_HI:[0-9]+]]]
; CI-DAG: s_load_dwordx2 s[0:1], s[4:5], 0x0

; GFX9: s_mov_b64 s[{{[0-9]+}}:[[HI:[0-9]+]]], src_shared_base
; GFX9: v_cmp_eq_u32_e32 vcc, s[[HI]], v[[PTR_HI]]

; CIT: v_cmp_eq_u32_e32 vcc, s4, v[[PTR_HI]]
; CIH: v_cmp_eq_u32_e32 vcc, s2, v[[PTR_HI]]
; GCN: v_cndmask_b32_e64 v{{[0-9]+}}, 0, 1, vcc
define amdgpu_kernel void @is_local_vgpr(ptr addrspace(1) %ptr.ptr) {
  %id = call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr inbounds ptr, ptr addrspace(1) %ptr.ptr, i32 %id
  %ptr = load volatile ptr, ptr addrspace(1) %gep
  %val = call i1 @llvm.amdgcn.is.shared(ptr %ptr)
  %ext = zext i1 %val to i32
  store i32 %ext, ptr addrspace(1) undef
  ret void
}

; FIXME: setcc (zero_extend (setcc)), 1) not folded out, resulting in
; select and vcc branch.

; GCN-LABEL: {{^}}is_local_sgpr:
; CI-DAG: s_load_dword s0, s[4:5], 0x1

; CI-DAG: s_load_dword [[PTR_HI:s[0-9]+]], s[4:5], 0x33{{$}}
; GFX9-DAG: s_load_dword [[PTR_HI:s[0-9]+]], s[4:5], 0x4{{$}}

; GFX9: s_mov_b64 s[{{[0-9]+}}:[[HI:[0-9]+]]], src_shared_base
; GFX9: s_cmp_eq_u32 [[PTR_HI]], s[[HI]]

; CI: s_cmp_eq_u32 s0, [[PTR_HI]]
; GCN: s_cbranch_vccnz
define amdgpu_kernel void @is_local_sgpr(ptr %ptr) {
  %val = call i1 @llvm.amdgcn.is.shared(ptr %ptr)
  br i1 %val, label %bb0, label %bb1

bb0:
  store volatile i32 0, ptr addrspace(1) undef
  br label %bb1

bb1:
  ret void
}

declare i32 @llvm.amdgcn.workitem.id.x() #0
declare i1 @llvm.amdgcn.is.shared(ptr nocapture) #0

attributes #0 = { nounwind readnone speculatable }

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"amdgpu_code_object_version", i32 500}
