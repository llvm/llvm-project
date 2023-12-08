; RUN: llc < %s -mtriple=i386-apple-darwin -mcpu=corei7 | FileCheck %s
; rdar://r7512579

; PHI defs in the atomic loop should be used by the add / adc
; instructions. They should not be dead.

define void @t(ptr nocapture %p) nounwind ssp {
entry:
; CHECK-LABEL: t:
; CHECK: movl ([[REG:%[a-z]+]]), %eax
; CHECK: movl 4([[REG]]), %edx
; CHECK: LBB0_1:
; CHECK: movl %eax, %ebx
; CHECK: addl $1, %ebx
; CHECK: movl %edx, %ecx
; CHECK: adcl $0, %ecx
; CHECK: lock cmpxchg8b ([[REG]])
; CHECK-NEXT: jne
  %0 = atomicrmw add ptr %p, i64 1 seq_cst
  ret void
}
