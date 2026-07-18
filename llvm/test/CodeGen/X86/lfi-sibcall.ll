; RUN: llc < %s -mtriple=x86_64_lfi | FileCheck %s

; LFI reserves %r11. For indirect vararg tail calls that consume all six
; argument GPRs, the function pointer would normally be loaded into %r11. With
; %r11 reserved there is no free register for the call target, so sibling-call
; optimization must be disabled.

define void @caller6_indirect_vararg(ptr %fn, i64 %a, i64 %b, i64 %c, i64 %d, i64 %e, i64 %f) {
; CHECK-LABEL: caller6_indirect_vararg:
; CHECK:       callq *
; CHECK-NOT:   TAILCALL
  tail call void (i64, i64, i64, i64, i64, i64, ...) %fn(i64 %a, i64 %b, i64 %c, i64 %d, i64 %e, i64 %f)
  ret void
}

define void @caller5_indirect_vararg(ptr %fn, i64 %a, i64 %b, i64 %c, i64 %d, i64 %e) {
; CHECK-LABEL: caller5_indirect_vararg:
; CHECK:       jmpq *{{.*}} # TAILCALL
  tail call void (i64, i64, i64, i64, i64, ...) %fn(i64 %a, i64 %b, i64 %c, i64 %d, i64 %e)
  ret void
}
