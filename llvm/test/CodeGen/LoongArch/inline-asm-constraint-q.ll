; RUN: llc --mtriple=loongarch32 --mattr=+f --verify-machineinstrs < %s | FileCheck %s
; RUN: llc --mtriple=loongarch64 --mattr=+f --verify-machineinstrs < %s | FileCheck %s

;; Check that the "q" operand is not R0.
define i32 @constraint_q_not_r0() {
; CHECK-NOT:    csrxchg ${{[a-z]*}}, $r0, 0
; CHECK-NOT:    csrxchg ${{[a-z]*}}, $zero, 0
entry:
  %2 = tail call i32 asm "csrxchg $0, $1, 0", "=r,q,0"(i32 0, i32 0)
  ret i32 %2
}

;; Check that the "q" operand is not R1.
define i32 @constraint_q_not_r1(i32 %0) {
; CHECK-NOT:    csrxchg ${{[a-z]*}}, $r1, 0
; CHECK-NOT:    csrxchg ${{[a-z]*}}, $ra, 0
entry:
  %2 = tail call i32 asm "", "={$r1},{$r1}"(i32 0)
  %3 = tail call i32 asm "csrxchg $0, $1, 0", "=r,q,0"(i32 %2, i32 %0)
  ret i32 %3
}
