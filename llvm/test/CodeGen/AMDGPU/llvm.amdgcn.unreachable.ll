; RUN: llc -mtriple=amdgcn %s -filetype=obj -filetype=null
; RUN: llc -global-isel=1 -new-reg-bank-select -mtriple=amdgcn %s -filetype=obj -filetype=null
; RUN: llc -mtriple=amdgcn < %s | FileCheck %s
; RUN: llc -global-isel=1 -new-reg-bank-select -mtriple=amdgcn < %s | FileCheck %s
define amdgpu_kernel void @f() {
  ; CHECK: ; divergent unreachable
  call void @llvm.amdgcn.unreachable()
  ret void
}

declare void @llvm.amdgcn.unreachable()
