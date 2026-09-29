; RUN: llc -mtriple=i686-unknown-linux-gnu -o - %s | FileCheck %s

declare void @f(i16 signext)
declare void @g(i32 signext)


define void @flags_match(i16 signext %x) {
entry:
  tail call void @f(i16 signext %x)
  ret void

; The parameter flags match; do the tail call.
; CHECK-LABEL: flags_match:
; CHECK: jmp f
}

define void @flags_mismatch(i16 zeroext %x) {
entry:
  tail call void @f(i16 signext %x)
  ret void

; The parameter flags mismatch. %x has not been sign-extended,
; so tail call is not possible.
; CHECK-LABEL: flags_mismatch:
; CHECK: movswl
; CHECK: calll f
}


define void @mismatch_doesnt_matter(i32 zeroext %x) {
entry:
  tail call void @g(i32 signext %x)
  ret void

; The parameter flags mismatch, but the type is wide enough that
; no extension takes place in practice, so do the tail call.

; CHECK-LABEL: mismatch_doesnt_matter:
; CHECK: jmp g
}
