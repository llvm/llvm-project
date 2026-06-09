; RUN: not llc -mtriple=amdgcn--amdpal -mcpu=gfx1100 -global-isel=0 < %s 2>&1 | FileCheck %s
; RUN: not llc -mtriple=amdgcn--amdpal -mcpu=gfx1100 -global-isel=1 < %s 2>&1 | FileCheck %s

declare amdgpu_cs_chain void @callee() nounwind

; The flags immarg must be 0 or 1. Any other value must be rejected rather than
; silently selecting the non-dynamic-VGPR path and dropping the trailing args.

; CHECK: in function test_invalid_flags void (): invalid flags value for amdgcn.cs.chain
define amdgpu_cs_chain void @test_invalid_flags() {
  call void(ptr, i32, i32, i32, i32, ...) @llvm.amdgcn.cs.chain(ptr @callee, i32 -1, i32 inreg 1, i32 2, i32 2, i32 inreg 32, i32 inreg -1, ptr @callee)
  unreachable
}
