; RUN: llc -mtriple=amdgcn %s -filetype=obj -filetype=null
; RUN: llc -mtriple=amdgcn < %s | FileCheck %s
define amdgpu_kernel void @f() {
  ; CHECK: ; divergent unreachable
  call void @llvm.amdgcn.unreachable()
  ret void
}

declare void @llvm.amdgcn.unreachable()
