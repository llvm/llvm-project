; RUN: llc -mtriple=x86_64-unknown-linux-gnu < %s | FileCheck %s

; The WinCall calling convention passes the fifth and later integer arguments
; in the APX extended GPRs R16-R19.  Any function that takes part in WinCall -
; as the callee itself or as the caller of a WinCall function - therefore has
; to turn on the egpr feature implicitly, even on a target whose triple does
; not enable APX.  Otherwise the physical register tables sized by
; X86RegisterInfo::getNumSupportedRegs() would not cover those registers and
; LiveVariables would abort on the live-in argument registers.
;
; This is reachable from ordinary C because the frontend accepts `wincall` on
; any x86-64 target, see clang/test/CodeGen/X86/wincall-sysv.c.

; The fifth argument is live in R16, which requires egpr to be enabled.
define x86_wincallcc i64 @callee5(i64 %a, i64 %b, i64 %c, i64 %d, i64 %e) {
entry:
  %s1 = add i64 %a, %b
  %s2 = add i64 %s1, %c
  %s3 = add i64 %s2, %d
  %s4 = add i64 %s3, %e
  ret i64 %s4
}
; CHECK-LABEL: callee5:
; CHECK: addq %rdx, %rcx
; CHECK: addq %r16

; The caller also has to materialize R16, so it needs egpr as well.
define i64 @caller5() {
entry:
  %r = call x86_wincallcc i64 @callee5(i64 1, i64 2, i64 3, i64 4, i64 5)
  ret i64 %r
}
; CHECK-LABEL: caller5:
; CHECK: movl $5, %r16d
