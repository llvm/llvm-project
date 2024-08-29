; RUN: llc < %s -mtriple=amdgcn -mcpu=gfx1030 -verify-machineinstrs | FileCheck %s -check-prefixes=GCN,PREGFX12
; RUN: llc < %s -mtriple=amdgcn -mcpu=gfx1031 -verify-machineinstrs | FileCheck %s -check-prefixes=GCN,PREGFX12
; RUN: llc < %s -mtriple=amdgcn -mcpu=gfx1200 -verify-machineinstrs | FileCheck %s -check-prefixes=GCN,GFX12PLUS

declare i32 @llvm.amdgcn.global.atomic.csub(ptr addrspace(1), i32)

; GCN-LABEL: {{^}}global_atomic_csub_rtn:
; PREGFX12: global_atomic_csub v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9:]+}}, s{{\[[0-9]+:[0-9]+\]}} glc
; GFX12PLUS: global_atomic_sub_clamp_u32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}, s{{\[[0-9]+:[0-9]+\]}} th:TH_ATOMIC_RETURN
define amdgpu_kernel void @global_atomic_csub_rtn(ptr addrspace(1) %ptr, i32 %data) {
main_body:
  %ret = call i32 @llvm.amdgcn.global.atomic.csub(ptr addrspace(1) %ptr, i32 %data)
  ret void
}

; GCN-LABEL: {{^}}global_atomic_csub_no_rtn:
; PREGFX12: global_atomic_csub v{{[0-9]+}}, v{{[0-9]+}}, s{{\[[0-9]+:[0-9]+\]}}
; GFX12PLUS: global_atomic_sub_clamp_u32 v{{[0-9]+}}, v{{[0-9]+}}, s{{\[[0-9]+:[0-9]+\]}}
define amdgpu_kernel void @global_atomic_csub_no_rtn(ptr addrspace(1) %ptr, i32 %data) #0 {
main_body:
  %ret = call i32 @llvm.amdgcn.global.atomic.csub(ptr addrspace(1) %ptr, i32 %data)
  ret void
}

; GCN-LABEL: {{^}}global_atomic_csub_off4_rtn:
; PREGFX12: global_atomic_csub v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}, s{{\[[0-9]+:[0-9]+\]}} offset:4 glc
; GFX12PLUS: global_atomic_sub_clamp_u32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}, s{{\[[0-9]+:[0-9]+\]}} offset:4 th:TH_ATOMIC_RETURN
define amdgpu_kernel void @global_atomic_csub_off4_rtn(ptr addrspace(1) %ptr, i32 %data) {
main_body:
  %p = getelementptr i32, ptr addrspace(1) %ptr, i64 1
  %ret = call i32 @llvm.amdgcn.global.atomic.csub(ptr addrspace(1) %p, i32 %data)
  ret void
}

; GCN-LABEL: {{^}}global_atomic_csub_off4_no_rtn:
; PREGFX12: global_atomic_csub v{{[0-9]+}}, v{{[0-9]+}}, s{{\[[0-9]+:[0-9]+\]}} offset:4
; GFX12PLUS: global_atomic_sub_clamp_u32 v{{[0-9]+}}, v{{[0-9]+}}, s{{\[[0-9]+:[0-9]+\]}} offset:4
define amdgpu_kernel void @global_atomic_csub_off4_no_rtn(ptr addrspace(1) %ptr, i32 %data) #0 {
main_body:
  %p = getelementptr i32, ptr addrspace(1) %ptr, i64 1
  %ret = call i32 @llvm.amdgcn.global.atomic.csub(ptr addrspace(1) %p, i32 %data)
  ret void
}

attributes #0 = { "target-features"="+atomic-csub-no-rtn-insts" }
