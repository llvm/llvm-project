; RUN: llc < %s -march=amdgcn -mcpu=gfx1030 -verify-machineinstrs | FileCheck %s -check-prefix=GCN
; RUN: llc < %s -march=amdgcn -mcpu=gfx1031 -verify-machineinstrs | FileCheck %s -check-prefix=GCN

declare i32 @llvm.amdgcn.buffer.atomic.csub(i32, <4 x i32>, i32, i32, i1)
declare i32 @llvm.amdgcn.global.atomic.csub(i32 addrspace(1)*, i32)

; GCN-LABEL: {{^}}buffer_atomic_csub:
; GCN: buffer_atomic_csub v0, v1, s[0:3], 0 idxen glc
define amdgpu_ps void @buffer_atomic_csub(<4 x i32> inreg %rsrc, i32 %data, i32 %vindex) {
main_body:
  %ret = call i32 @llvm.amdgcn.buffer.atomic.csub(i32 %data, <4 x i32> %rsrc, i32 %vindex, i32 0, i1 0)
  ret void
}

; GCN-LABEL: {{^}}buffer_atomic_csub_off4_slc:
; GCN: buffer_atomic_csub v0, v1, s[0:3], 0 idxen offset:4 glc slc
define amdgpu_ps void @buffer_atomic_csub_off4_slc(<4 x i32> inreg %rsrc, i32 %data, i32 %vindex) {
main_body:
  %ret = call i32 @llvm.amdgcn.buffer.atomic.csub(i32 %data, <4 x i32> %rsrc, i32 %vindex, i32 4, i1 1)
  ret void
}

; GCN-LABEL: {{^}}global_atomic_csub:
; GCN: global_atomic_csub v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9:]+}}, s{{\[[0-9]+:[0-9]+\]}} glc
define amdgpu_kernel void @global_atomic_csub(i32 addrspace(1)* %ptr, i32 %data) {
main_body:
  %ret = call i32 @llvm.amdgcn.global.atomic.csub(i32 addrspace(1)* %ptr, i32 %data)
  ret void
}

; GCN-LABEL: {{^}}global_atomic_csub_off4:
; GCN: global_atomic_csub v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}, s{{\[[0-9]+:[0-9]+\]}} offset:4 glc
define amdgpu_kernel void @global_atomic_csub_off4(i32 addrspace(1)* %ptr, i32 %data) {
main_body:
  %p = getelementptr i32, i32 addrspace(1)* %ptr, i64 1
  %ret = call i32 @llvm.amdgcn.global.atomic.csub(i32 addrspace(1)* %p, i32 %data)
  ret void
}
