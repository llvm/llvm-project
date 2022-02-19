; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=hawaii -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GCN,CI %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GCN,GFX9 %s

; GCN-LABEL: {{^}}is_local_vgpr:
; GCN-DAG: {{flat|global}}_load_dwordx2 v{{\[[0-9]+}}:[[PTR_HI:[0-9]+]]]
; CI-DAG: s_load_dword [[APERTURE:s[0-9]+]], s[4:5], 0x10
; GFX9-DAG: s_getreg_b32 [[APERTURE:s[0-9]+]], hwreg(HW_REG_SH_MEM_BASES, 16, 16)
; GFX9: s_lshl_b32 [[APERTURE]], [[APERTURE]], 16

; GCN: v_cmp_eq_u32_e32 vcc, [[APERTURE]], v[[PTR_HI]]
; GCN: v_cndmask_b32_e64 v{{[0-9]+}}, 0, 1, vcc
define amdgpu_kernel void @is_local_vgpr(i8* addrspace(1)* %ptr.ptr) {
  %id = call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr inbounds i8*, i8* addrspace(1)* %ptr.ptr, i32 %id
  %ptr = load volatile i8*, i8* addrspace(1)* %gep
  %val = call i1 @llvm.amdgcn.is.shared(i8* %ptr)
  %ext = zext i1 %val to i32
  store i32 %ext, i32 addrspace(1)* undef
  ret void
}

; FIXME: setcc (zero_extend (setcc)), 1) not folded out, resulting in
; select and vcc branch.

; GCN-LABEL: {{^}}is_local_sgpr:
; CI-DAG: s_load_dword [[APERTURE:s[0-9]+]], s[4:5], 0x10{{$}}
; GFX9-DAG: s_getreg_b32 [[APERTURE:s[0-9]+]], hwreg(HW_REG_SH_MEM_BASES, 16, 16)
; GFX9-DAG: s_lshl_b32 [[APERTURE]], [[APERTURE]], 16

; CI-DAG: s_load_dword [[PTR_HI:s[0-9]+]], s[6:7], 0x1{{$}}
; GFX9-DAG: s_load_dword [[PTR_HI:s[0-9]+]], s[6:7], 0x4{{$}}

; GCN: s_cmp_eq_u32 [[PTR_HI]], [[APERTURE]]
; GCN: s_cbranch_vccnz
define amdgpu_kernel void @is_local_sgpr(i8* %ptr) {
  %val = call i1 @llvm.amdgcn.is.shared(i8* %ptr)
  br i1 %val, label %bb0, label %bb1

bb0:
  store volatile i32 0, i32 addrspace(1)* undef
  br label %bb1

bb1:
  ret void
}

declare i32 @llvm.amdgcn.workitem.id.x() #0
declare i1 @llvm.amdgcn.is.shared(i8* nocapture) #0

attributes #0 = { nounwind readnone speculatable }
