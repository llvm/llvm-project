; RUN: llc -mtriple=amdgpu6.00 %s -filetype=obj -filetype=null
; RUN: llc -global-isel=1 -mtriple=amdgpu6.00 %s -filetype=obj -filetype=null
; RUN: llc -mtriple=amdgpu6.00 < %s | FileCheck %s
; RUN: llc -global-isel=1 -mtriple=amdgpu6.00 < %s | FileCheck %s
define amdgpu_kernel void @f() {
  ; CHECK: ; divergent unreachable
  call void @llvm.amdgcn.unreachable()
  ret void
}

declare void @llvm.amdgcn.unreachable()
