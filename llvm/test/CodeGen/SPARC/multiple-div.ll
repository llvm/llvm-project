; RUN: llc < %s -mtriple=sparc | FileCheck %s
; RUN: llc -O0 < %s -mtriple=sparc | FileCheck %s

;; llc -O0 used to try to spill Y to the stack, which isn't possible,
;; and then crashed. Additionally, in -O1, it would omit the second
;; apparently-redundant wr to %y, which is not actually redundant
;; because the spec says to treat %y as potentially-written by udiv.

; CHECK-LABEL: two_divides:
; CHECK: wr %g0, %g0, %y
; CHECK: udiv
; CHECK: wr %g0, %g0, %y
; CHECK: udiv
; CHECK: add

define i32 @two_divides(i32 %a, i32 %b) {
  %r = udiv i32 %a, %b
  %r2 = udiv i32 %b, %a
  %r3 = add i32 %r, %r2
  ret i32 %r3
}
