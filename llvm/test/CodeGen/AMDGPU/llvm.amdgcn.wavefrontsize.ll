; RUN: llc -mtriple=amdgcn -mcpu=fiji -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,W64 %s
; RUN: llc -mtriple=amdgcn -mcpu=gfx1010 -mattr=+wavefrontsize32 -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,W32 %s
; RUN: llc -mtriple=amdgcn -mcpu=gfx1010 -mattr=+wavefrontsize64 -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,W64 %s
; RUN: llc -mtriple=amdgcn -mcpu=gfx1100 -mattr=+wavefrontsize32 -verify-machineinstrs -amdgpu-enable-vopd=0 < %s | FileCheck -check-prefixes=GCN,W32 %s
; RUN: llc -mtriple=amdgcn -mcpu=gfx1100 -mattr=+wavefrontsize64 -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,W64 %s

; GCN-LABEL: {{^}}fold_wavefrontsize:

; W32:       v_mov_b32_e32 [[V:v[0-9]+]], 32
; W64:       v_mov_b32_e32 [[V:v[0-9]+]], 64
; GCN:       store_{{dword|b32}} v{{.+}}, [[V]]


define amdgpu_kernel void @fold_wavefrontsize(ptr addrspace(1) nocapture %arg) {

bb:
  %tmp = tail call i32 @llvm.amdgcn.wavefrontsize() #0
  store i32 %tmp, ptr addrspace(1) %arg, align 4
  ret void
}

; GCN-LABEL: {{^}}fold_and_optimize_wavefrontsize:

; W32:       v_mov_b32_e32 [[V:v[0-9]+]], 1{{$}}
; W64:       v_mov_b32_e32 [[V:v[0-9]+]], 2{{$}}
; GCN-NOT:   cndmask
; GCN:       store_{{dword|b32}} v{{.+}}, [[V]]


define amdgpu_kernel void @fold_and_optimize_wavefrontsize(ptr addrspace(1) nocapture %arg) {
bb:
  %tmp = tail call i32 @llvm.amdgcn.wavefrontsize() #0
  %tmp1 = icmp ugt i32 %tmp, 32
  %tmp2 = select i1 %tmp1, i32 2, i32 1
  store i32 %tmp2, ptr addrspace(1) %arg
  ret void
}

; GCN-LABEL: {{^}}fold_and_optimize_if_wavefrontsize:

define amdgpu_kernel void @fold_and_optimize_if_wavefrontsize(ptr addrspace(1) nocapture %arg) {
bb:
  %tmp = tail call i32 @llvm.amdgcn.wavefrontsize() #0
  %tmp1 = icmp ugt i32 %tmp, 32
  br i1 %tmp1, label %bb2, label %bb3

bb2:                                              ; preds = %bb
  store i32 1, ptr addrspace(1) %arg, align 4
  br label %bb3

bb3:                                              ; preds = %bb2, %bb
  ret void
}

declare i32 @llvm.amdgcn.wavefrontsize() #0

attributes #0 = { nounwind readnone speculatable }
