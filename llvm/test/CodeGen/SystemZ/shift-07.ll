; Test 32-bit arithmetic shifts right.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

; Check the low end of the SRAG range.
define i64 @f1(i64 %a) {
; CHECK-LABEL: f1:
; CHECK: srag %r2, %r2, 1
; CHECK: br %r14
  %shift = ashr i64 %a, 1
  ret i64 %shift
}

; Check the high end of the defined SRAG range.
define i64 @f2(i64 %a) {
; CHECK-LABEL: f2:
; CHECK: srag %r2, %r2, 63
; CHECK: br %r14
  %shift = ashr i64 %a, 63
  ret i64 %shift
}

; We don't generate shifts by out-of-range values.
define i64 @f3(i64 %a) {
; CHECK-LABEL: f3:
; CHECK-NOT: srag
; CHECK: br %r14
  %shift = ashr i64 %a, 64
  ret i64 %shift
}

; Check variable shifts.
define i64 @f4(i64 %a, i64 %amt) {
; CHECK-LABEL: f4:
; CHECK: srag %r2, %r2, 0(%r3)
; CHECK: br %r14
  %shift = ashr i64 %a, %amt
  ret i64 %shift
}

; Check shift amounts that have a constant term.
define i64 @f5(i64 %a, i64 %amt) {
; CHECK-LABEL: f5:
; CHECK: srag %r2, %r2, 10(%r3)
; CHECK: br %r14
  %add = add i64 %amt, 10
  %shift = ashr i64 %a, %add
  ret i64 %shift
}

; ...and again with a sign-extended 32-bit shift amount.
define i64 @f6(i64 %a, i32 %amt) {
; CHECK-LABEL: f6:
; CHECK: srag %r2, %r2, 10(%r3)
; CHECK: br %r14
  %add = add i32 %amt, 10
  %addext = sext i32 %add to i64
  %shift = ashr i64 %a, %addext
  ret i64 %shift
}

; ...and now with a zero-extended 32-bit shift amount.
define i64 @f7(i64 %a, i32 %amt) {
; CHECK-LABEL: f7:
; CHECK: srag %r2, %r2, 10(%r3)
; CHECK: br %r14
  %add = add i32 %amt, 10
  %addext = zext i32 %add to i64
  %shift = ashr i64 %a, %addext
  ret i64 %shift
}

; Check shift amounts that have the largest in-range constant term.  We could
; mask the amount instead.
define i64 @f8(i64 %a, i64 %amt) {
; CHECK-LABEL: f8:
; CHECK: srag %r2, %r2, 524287(%r3)
; CHECK: br %r14
  %add = add i64 %amt, 524287
  %shift = ashr i64 %a, %add
  ret i64 %shift
}

; Check the next value up, which without masking must use a separate
; addition.
define i64 @f9(i64 %a, i64 %amt) {
; CHECK-LABEL: f9:
; CHECK: a{{g?}}fi %r3, 524288
; CHECK: srag %r2, %r2, 0(%r3)
; CHECK: br %r14
  %add = add i64 %amt, 524288
  %shift = ashr i64 %a, %add
  ret i64 %shift
}

; Check cases where 1 is subtracted from the shift amount.
define i64 @f10(i64 %a, i64 %amt) {
; CHECK-LABEL: f10:
; CHECK: srag %r2, %r2, -1(%r3)
; CHECK: br %r14
  %sub = sub i64 %amt, 1
  %shift = ashr i64 %a, %sub
  ret i64 %shift
}

; Check the lowest value that can be subtracted from the shift amount.
; Again, we could mask the shift amount instead.
define i64 @f11(i64 %a, i64 %amt) {
; CHECK-LABEL: f11:
; CHECK: srag %r2, %r2, -524288(%r3)
; CHECK: br %r14
  %sub = sub i64 %amt, 524288
  %shift = ashr i64 %a, %sub
  ret i64 %shift
}

; Check the next value down, which without masking must use a separate
; addition.
define i64 @f12(i64 %a, i64 %amt) {
; CHECK-LABEL: f12:
; CHECK: a{{g?}}fi %r3, -524289
; CHECK: srag %r2, %r2, 0(%r3)
; CHECK: br %r14
  %sub = sub i64 %amt, 524289
  %shift = ashr i64 %a, %sub
  ret i64 %shift
}

; Check that we don't try to generate "indexed" shifts.
define i64 @f13(i64 %a, i64 %b, i64 %c) {
; CHECK-LABEL: f13:
; CHECK: a{{g?}}r {{%r3, %r4|%r4, %r3}}
; CHECK: srag %r2, %r2, 0({{%r[34]}})
; CHECK: br %r14
  %add = add i64 %b, %c
  %shift = ashr i64 %a, %add
  ret i64 %shift
}

; Check that the shift amount uses an address register.  It cannot be in %r0.
define i64 @f14(i64 %a, i64 *%ptr) {
; CHECK-LABEL: f14:
; CHECK: l %r1, 4(%r3)
; CHECK: srag %r2, %r2, 0(%r1)
; CHECK: br %r14
  %amt = load i64, i64 *%ptr
  %shift = ashr i64 %a, %amt
  ret i64 %shift
}
