; RUN: not llc -mtriple=amdgcn--amdpal -mcpu=gfx1200 -global-isel=1 -mattr=+wavefrontsize64 -verify-machineinstrs=0 < %s 2>&1 | FileCheck %s
; RUN: not llc -mtriple=amdgcn--amdpal -mcpu=gfx1200 -global-isel=0 -mattr=+wavefrontsize64 -verify-machineinstrs=0 < %s 2>&1 | FileCheck %s

declare amdgpu_cs_chain void @callee() nounwind

; CHECK: in function test_dvgpr void (): dynamic VGPR mode is only supported for wave32
define amdgpu_cs_chain void @test_dvgpr() {
  call void(ptr, i64, i32, i32, i32, ...) @llvm.amdgcn.cs.chain(ptr @callee, i64 -1, i32 inreg 1, i32 2, i32 1, i32 inreg 32, i32 inreg -1, ptr @callee)
  unreachable
}

