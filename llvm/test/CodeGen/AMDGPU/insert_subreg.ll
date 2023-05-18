; RUN: llc -mtriple=amdgcn -mcpu=tahiti -mattr=-promote-alloca -verify-machineinstrs < %s
; RUN: llc -mtriple=amdgcn -mcpu=tonga -mattr=-promote-alloca -verify-machineinstrs < %s

; Test that INSERT_SUBREG instructions don't have non-register operands after
; instruction selection.

; Make sure this doesn't crash
; CHECK-LABEL: test:
define amdgpu_kernel void @test(ptr addrspace(1) %out) {
entry:
  %tmp0 = alloca [16 x i32], addrspace(5)
  %tmp1 = ptrtoint ptr addrspace(5) %tmp0 to i32
  %tmp2 = sext i32 %tmp1 to i64
  store i64 %tmp2, ptr addrspace(1) %out
  ret void
}
