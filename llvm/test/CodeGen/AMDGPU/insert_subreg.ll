; RUN: llc -march=amdgcn -mcpu=SI -mattr=-promote-alloca -verify-machineinstrs < %s
; RUN: llc -march=amdgcn -mcpu=tonga -mattr=-promote-alloca -verify-machineinstrs < %s

; Test that INSERT_SUBREG instructions don't have non-register operands after
; instruction selection.

; Make sure this doesn't crash
; CHECK-LABEL: test:
define amdgpu_kernel void @test(i64 addrspace(1)* %out) {
entry:
  %tmp0 = alloca [16 x i32]
  %tmp1 = ptrtoint [16 x i32]* %tmp0 to i32
  %tmp2 = sext i32 %tmp1 to i64
  store i64 %tmp2, i64 addrspace(1)* %out
  ret void
}
