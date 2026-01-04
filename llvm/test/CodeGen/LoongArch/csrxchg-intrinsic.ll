; RUN: llc --mtriple=loongarch32 --mattr=+f --verify-machineinstrs < %s | FileCheck %s
; RUN: llc --mtriple=loongarch64 --mattr=+f --verify-machineinstrs < %s | FileCheck %s

declare i32 @llvm.loongarch.csrxchg.w(i32, i32, i32 immarg)

;; Check that the rj operand of csrxchg is not R0.
define void @csrxchg_w_rj_not_r0(i32 signext %a) {
; CHECK-NOT:    csrxchg ${{[a-z]*}}, $r0, 0
; CHECK-NOT:    csrxchg ${{[a-z]*}}, $zero, 0
entry:
  %0 = tail call i32 @llvm.loongarch.csrxchg.w(i32 %a, i32 0, i32 0)
  ret void
}

;; Check that the rj operand of csrxchg is not R1.
define i32 @csrxchg_w_rj_not_r1(i32 %0) {
; CHECK-NOT:    csrxchg ${{[a-z]*}}, $r1, 0
; CHECK-NOT:    csrxchg ${{[a-z]*}}, $ra, 0
entry:
  %2 = tail call i32 asm "", "={$r1},{$r1}"(i32 0)
  %3 = tail call i32 @llvm.loongarch.csrxchg.w(i32 %0, i32 %2, i32 0)
  ret i32 %3
}
