; RUN: split-file %s %t
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -global-isel -O3 -verify-machineinstrs < %t/constants.ll | FileCheck %s --check-prefix=CHECK
; RUN: not llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -global-isel -O3 -verify-machineinstrs < %t/nonconstant.ll 2>&1 | FileCheck %s --check-prefix=ERR

;--- constants.ll
declare i32 @llvm.amdgcn.ballot.i32(i1)
declare i32 @llvm.amdgcn.mbcnt.lo(i32, i32)

define i32 @ballot_false_wave64() {
; CHECK-LABEL: ballot_false_wave64:
; CHECK:       v_mov_b32_e32 v0, 0
  %ballot = call i32 @llvm.amdgcn.ballot.i32(i1 false)
  ret i32 %ballot
}

define i32 @ballot_true_wave64() {
; CHECK-LABEL: ballot_true_wave64:
; CHECK:       v_mov_b32_e32 v0, exec_lo
  %ballot = call i32 @llvm.amdgcn.ballot.i32(i1 true)
  ret i32 %ballot
}

define i32 @activelane_false() {
; CHECK-LABEL: activelane_false:
; CHECK:       v_mbcnt_lo_u32_b32 v0, 0, 0
  %ballot = call i32 @llvm.amdgcn.ballot.i32(i1 false)
  %lane = call i32 @llvm.amdgcn.mbcnt.lo(i32 %ballot, i32 0)
  ret i32 %lane
}

;--- nonconstant.ll
declare i32 @llvm.amdgcn.ballot.i32(i1)

define amdgpu_cs i32 @nonconstant_i32_ballot_wave64(i32 %x) {
; ERR: LLVM ERROR: cannot select: %{{[0-9]+}}:sreg_32(s32) = G_INTRINSIC_CONVERGENT intrinsic(@llvm.amdgcn.ballot)
  %trunc = trunc i32 %x to i1
  %ballot = call i32 @llvm.amdgcn.ballot.i32(i1 %trunc)
  ret i32 %ballot
}
