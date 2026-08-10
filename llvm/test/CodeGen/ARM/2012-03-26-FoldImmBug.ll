; RUN: llc < %s -mtriple=thumbv7-apple-ios -mcpu=cortex-a8 | FileCheck %s

; ARM has a peephole optimization which looks for a def / use pair. The def
; produces a 32-bit immediate which is consumed by the use. It tries to 
; fold the immediate by breaking it into two parts and fold them into the
; immmediate fields of two uses. e.g
;        movw    r2, #40885
;        movt    r3, #46540
;        add     r0, r0, r3
; =>
;        add.w   r0, r0, #3019898880
;        add.w   r0, r0, #30146560
;
; However, this transformation is incorrect if the user produces a flag. e.g.
;        movw    r2, #40885
;        movt    r3, #46540
;        adds    r0, r0, r3
; =>
;        add.w   r0, r0, #3019898880
;        adds.w  r0, r0, #30146560
; Note the adds.w may not set the carry flag even if the original sequence
; would.
;
; rdar://11116189
define i64 @t(i64 %aInput) nounwind {
; CHECK-LABEL: t:
; CHECK: movs [[REG:(r[0-9]+)]], #0
; CHECK: movt [[REG]], #46540
; CHECK: adds r{{[0-9]+}}, r{{[0-9]+}}, [[REG]]
  %1 = mul i64 %aInput, 1000000
  %2 = add i64 %1, -7952618389194932224
  ret i64 %2
}
