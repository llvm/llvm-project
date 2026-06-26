; RUN: llc -mtriple=amdgcn -mcpu=gfx600 %s -filetype=obj -filetype=null
; RUN: llc -global-isel=1 -mtriple=amdgcn -mcpu=gfx600 %s -filetype=obj -filetype=null
; RUN: llc -mtriple=amdgcn -mcpu=gfx600 < %s | FileCheck %s
; RUN: llc -global-isel=1 -mtriple=amdgcn -mcpu=gfx600 < %s | FileCheck %s
define amdgpu_kernel void @f() {
  ; CHECK: ; divergent unreachable
  call void @llvm.amdgcn.unreachable()
  ret void
}

declare void @llvm.amdgcn.unreachable()
