; RUN: llc < %s -mtriple=amdgcn -mcpu=gfx1030 -verify-machineinstrs | FileCheck %s -check-prefixes=GCN,PREGFX12
; RUN: llc < %s -mtriple=amdgcn -mcpu=gfx1031 -verify-machineinstrs | FileCheck %s -check-prefixes=GCN,PREGFX12
; RUN: llc < %s -mtriple=amdgcn -mcpu=gfx1200 -verify-machineinstrs | FileCheck %s -check-prefixes=GCN,GFX12PLUS

declare i32 @llvm.amdgcn.buffer.atomic.csub(i32, <4 x i32>, i32, i32, i1)
declare i32 @llvm.amdgcn.global.atomic.csub(ptr addrspace(1), i32)

; GCN-LABEL: {{^}}buffer_atomic_csub_rtn:
; PREGFX12: buffer_atomic_csub v0, v1, s[0:3], 0 idxen glc
; GFX12PLUS: buffer_atomic_sub_clamp_u32 v0, v1, s[0:3], null idxen th:TH_ATOMIC_RETURN
define amdgpu_ps void @buffer_atomic_csub_rtn(<4 x i32> inreg %rsrc, i32 %data, i32 %vindex) {
main_body:
  %ret = call i32 @llvm.amdgcn.buffer.atomic.csub(i32 %data, <4 x i32> %rsrc, i32 %vindex, i32 0, i1 0)
  ret void
}

; GCN-LABEL: {{^}}buffer_atomic_csub_no_rtn:
; PREGFX12: buffer_atomic_csub v0, v1, s[0:3], 0 idxen
; GFX12PLUS: buffer_atomic_csub_u32 v0, v1, s[0:3], null idxen
define amdgpu_ps void @buffer_atomic_csub_no_rtn(<4 x i32> inreg %rsrc, i32 %data, i32 %vindex) #0 {
main_body:
  %ret = call i32 @llvm.amdgcn.buffer.atomic.csub(i32 %data, <4 x i32> %rsrc, i32 %vindex, i32 0, i1 0)
  ret void
}

; GCN-LABEL: {{^}}buffer_atomic_csub_off4_slc_rtn:
; PREGFX12: buffer_atomic_csub v0, v1, s[0:3], 0 idxen offset:4 glc slc
; GFX12PLUS: buffer_atomic_sub_clamp_u32 v0, v1, s[0:3], null idxen offset:4 th:TH_ATOMIC_NT_RETURN
define amdgpu_ps void @buffer_atomic_csub_off4_slc_rtn(<4 x i32> inreg %rsrc, i32 %data, i32 %vindex) {
main_body:
  %ret = call i32 @llvm.amdgcn.buffer.atomic.csub(i32 %data, <4 x i32> %rsrc, i32 %vindex, i32 4, i1 1)
  ret void
}

; GCN-LABEL: {{^}}buffer_atomic_csub_off4_slc_no_rtn:
; PREGFX12: buffer_atomic_csub v0, v1, s[0:3], 0 idxen offset:4 slc
; GFX12PLUS: buffer_atomic_csub_u32 v0, v1, s[0:3], null idxen offset:4 th:TH_ATOMIC_NT
define amdgpu_ps void @buffer_atomic_csub_off4_slc_no_rtn(<4 x i32> inreg %rsrc, i32 %data, i32 %vindex) #0 {
main_body:
  %ret = call i32 @llvm.amdgcn.buffer.atomic.csub(i32 %data, <4 x i32> %rsrc, i32 %vindex, i32 4, i1 1)
  ret void
}

; GCN-LABEL: {{^}}global_atomic_csub_rtn:
; PREGFX12: global_atomic_csub v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9:]+}}, s{{\[[0-9]+:[0-9]+\]}} glc
; GFX12PLUS: global_atomic_sub_clamp_u32 v0, v0, v1, s[0:1] th:TH_ATOMIC_RETURN
define amdgpu_kernel void @global_atomic_csub_rtn(ptr addrspace(1) %ptr, i32 %data) {
main_body:
  %ret = call i32 @llvm.amdgcn.global.atomic.csub(ptr addrspace(1) %ptr, i32 %data)
  ret void
}

; GCN-LABEL: {{^}}global_atomic_csub_no_rtn:
; PREGFX12: global_atomic_csub v{{[0-9]+}}, v{{[0-9]+}}, s{{\[[0-9]+:[0-9]+\]}}
; GFX12PLUS: global_atomic_sub_clamp_u32 v0, v1, s[0:1]
define amdgpu_kernel void @global_atomic_csub_no_rtn(ptr addrspace(1) %ptr, i32 %data) #0 {
main_body:
  %ret = call i32 @llvm.amdgcn.global.atomic.csub(ptr addrspace(1) %ptr, i32 %data)
  ret void
}

; GCN-LABEL: {{^}}global_atomic_csub_off4_rtn:
; PREGFX12: global_atomic_csub v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}, s{{\[[0-9]+:[0-9]+\]}} offset:4 glc
; GFX12PLUS: global_atomic_sub_clamp_u32 v0, v0, v1, s[0:1] offset:4 th:TH_ATOMIC_RETURN
define amdgpu_kernel void @global_atomic_csub_off4_rtn(ptr addrspace(1) %ptr, i32 %data) {
main_body:
  %p = getelementptr i32, ptr addrspace(1) %ptr, i64 1
  %ret = call i32 @llvm.amdgcn.global.atomic.csub(ptr addrspace(1) %p, i32 %data)
  ret void
}

; GCN-LABEL: {{^}}global_atomic_csub_off4_no_rtn:
; PREGFX12: global_atomic_csub v{{[0-9]+}}, v{{[0-9]+}}, s{{\[[0-9]+:[0-9]+\]}} offset:4
; GFX12PLUS: global_atomic_sub_clamp_u32 v0, v1, s[0:1] offset:4
define amdgpu_kernel void @global_atomic_csub_off4_no_rtn(ptr addrspace(1) %ptr, i32 %data) #0 {
main_body:
  %p = getelementptr i32, ptr addrspace(1) %ptr, i64 1
  %ret = call i32 @llvm.amdgcn.global.atomic.csub(ptr addrspace(1) %p, i32 %data)
  ret void
}

attributes #0 = { "target-features"="+atomic-csub-no-rtn-insts" }
