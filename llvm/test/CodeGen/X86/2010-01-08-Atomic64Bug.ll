; RUN: llc < %s -mtriple=i386-apple-darwin -mcpu=corei7 | FileCheck %s
; rdar://r7512579

; PHI defs in the atomic loop should be used by the add / adc
; instructions. They should not be dead.

define void @t(i64* nocapture %p) nounwind ssp {
entry:
; CHECK: t:
; CHECK: movl ([[REG:%[a-z]+]]), %eax
; CHECK: movl 4([[REG]]), %edx
; CHECK: LBB0_1:
; CHECK: movl $1
; CHECK: addl
; CHECK: movl $0
; CHECK: adcl
; CHECK: lock
; CHECK-NEXT: cmpxchg8b ([[REG]])
; CHECK-NEXT: jne
  %0 = atomicrmw add i64* %p, i64 1 seq_cst
  ret void
}
