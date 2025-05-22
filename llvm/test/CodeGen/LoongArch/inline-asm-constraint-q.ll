; RUN: llc --mtriple=loongarch32 --mattr=+f --verify-machineinstrs < %s | FileCheck %s
; RUN: llc --mtriple=loongarch64 --mattr=+f --verify-machineinstrs < %s | FileCheck %s

;; Check that the "q" operand is not R0.
define i32 @constraint_q_not_r0(i32 %a) {
; CHECK-NOT:    csrxchg ${{[a-z]*}}, $r0, 0
; CHECK-NOT:    csrxchg ${{[a-z]*}}, $zero, 0
entry:
  %1 = tail call i32 asm "csrxchg $0, $1, $2", "=r,q,i,0"(i32 0, i32 0, i32 %a)
  ret i32 %1
}

;; Check that the "q" operand is not R1.
define i32 @constraint_q_not_r1() {
; CHECK-NOT:    csrxchg ${{[a-z]*}}, $r1, 0
; CHECK-NOT:    csrxchg ${{[a-z]*}}, $ra, 0
entry:
  %0 = tail call i32 asm "csrxchg $0, $1, $2", "=r,q,i,{r4},{r5},{r6},{r7},{r8},{r9},{r10},{r11},{r12},{r13},{r14},{r15},{r16},{r17},{r18},{r19},{r20},{r23},{r24},{r25},{r26},{r27},{r28},{r29},{r30},{r31},0"(i32 4, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)
  ret i32 %0
}
