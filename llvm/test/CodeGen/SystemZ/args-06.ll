; Test the padding of unextended integer stack parameters.  These are used
; to pass structures.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

define i8 @f1(i8 %a, i8 %b, i8 %c, i8 %d, i8 %e, i8 %f, i8 %g) {
; CHECK-LABEL: f1:
; CHECK: lb {{%r[0-5]}}, 175(%r15)
; CHECK: lb {{%r[0-5]}}, 167(%r15)
; CHECK: ar %r2, %r3
; CHECK: ar %r2, %r4
; CHECK: ar %r2, %r5
; CHECK: ar %r2, %r6
; CHECK: br %r14
  %addb = add i8 %a, %b
  %addc = add i8 %addb, %c
  %addd = add i8 %addc, %d
  %adde = add i8 %addd, %e
  %addf = add i8 %adde, %f
  %addg = add i8 %addf, %g
  ret i8 %addg
}

define i16 @f2(i16 %a, i16 %b, i16 %c, i16 %d, i16 %e, i16 %f, i16 %g) {
; CHECK-LABEL: f2:
; CHECK: ar %r2, %r3
; CHECK: ar %r2, %r4
; CHECK: ar %r2, %r5
; CHECK: ar %r2, %r6
; CHECK: ah %r2, 166(%r15)
; CHECK: ah %r2, 174(%r15)
; CHECK: br %r14
  %addb = add i16 %a, %b
  %addc = add i16 %addb, %c
  %addd = add i16 %addc, %d
  %adde = add i16 %addd, %e
  %addf = add i16 %adde, %f
  %addg = add i16 %addf, %g
  ret i16 %addg
}

define i32 @f3(i32 %a, i32 %b, i32 %c, i32 %d, i32 %e, i32 %f, i32 %g) {
; CHECK-LABEL: f3:
; CHECK: ar %r2, %r3
; CHECK: ar %r2, %r4
; CHECK: ar %r2, %r5
; CHECK: ar %r2, %r6
; CHECK: a %r2, 164(%r15)
; CHECK: a %r2, 172(%r15)
; CHECK: br %r14
  %addb = add i32 %a, %b
  %addc = add i32 %addb, %c
  %addd = add i32 %addc, %d
  %adde = add i32 %addd, %e
  %addf = add i32 %adde, %f
  %addg = add i32 %addf, %g
  ret i32 %addg
}

define i64 @f4(i64 %a, i64 %b, i64 %c, i64 %d, i64 %e, i64 %f, i64 %g) {
; CHECK-LABEL: f4:
; CHECK: agr %r2, %r3
; CHECK: agr %r2, %r4
; CHECK: agr %r2, %r5
; CHECK: agr %r2, %r6
; CHECK: ag %r2, 160(%r15)
; CHECK: ag %r2, 168(%r15)
; CHECK: br %r14
  %addb = add i64 %a, %b
  %addc = add i64 %addb, %c
  %addd = add i64 %addc, %d
  %adde = add i64 %addd, %e
  %addf = add i64 %adde, %f
  %addg = add i64 %addf, %g
  ret i64 %addg
}
