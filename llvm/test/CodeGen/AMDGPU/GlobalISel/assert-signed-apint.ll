; REQUIRES: asserts
; RUN: llc -global-isel -mtriple=amdgcn -mcpu=gfx1200 -verify-machineinstrs -stop-after=instruction-select -o - %s | FileCheck %s

; Check that APInt doesn't assert on creation from -2147483648 value.

; CHECK-LABEL: @test
; CHECK: S_BUFFER_LOAD_DWORD_SGPR_IMM

define amdgpu_cs void @test(<4 x i32> inreg %base, i32 inreg %i, ptr addrspace(1) inreg %out) {
  %off = or i32 %i, -2147483648
  %v = call i32 @llvm.amdgcn.s.buffer.load.i32(<4 x i32> %base, i32 %off, i32 0)
  store i32 %v, ptr addrspace(1) %out, align 4
  ret void
}
