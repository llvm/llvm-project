; RUN: llc -mtriple=x86_64-pc-windows-msvc %s -o - | FileCheck %s

; Win64 passes byval and vector arguments indirectly, through a temporary in
; the caller's frame, which a guaranteed tail call would discard.  A call that
; isn't required to be a tail call becomes an ordinary call.

declare tailcc void @byval_callee(i64, ptr byval([9 x i64]) align 8)
declare tailcc void @vec_callee(<4 x float>)

define tailcc void @byval_caller(i64 %a, ptr byval([9 x i64]) align 8 %p) {
; CHECK-LABEL: byval_caller:
; CHECK:         callq byval_callee
  tail call tailcc void @byval_callee(i64 %a, ptr byval([9 x i64]) align 8 %p)
  ret void
}

define tailcc void @vec_caller(<4 x float> %x) {
; CHECK-LABEL: vec_caller:
; CHECK:         callq vec_callee
  tail call tailcc void @vec_callee(<4 x float> %x)
  ret void
}
