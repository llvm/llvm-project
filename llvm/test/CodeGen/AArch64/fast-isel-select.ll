; RUN: llc -mtriple=aarch64-apple-darwin                             -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=aarch64-apple-darwin -fast-isel -fast-isel-abort=1 -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=aarch64-apple-darwin -global-isel -verify-machineinstrs < %s | FileCheck %s --check-prefix=GISEL

; First test the different supported value types for select.
define zeroext i1 @select_i1(i1 zeroext %c, i1 zeroext %a, i1 zeroext %b) {
; CHECK-LABEL: select_i1
; CHECK:       {{cmp w0, #0|tst w0, #0x1}}
; CHECK-NEXT:  csel {{w[0-9]+}}, w1, w2, ne
  %1 = select i1 %c, i1 %a, i1 %b
  ret i1 %1
}

define zeroext i8 @select_i8(i1 zeroext %c, i8 zeroext %a, i8 zeroext %b) {
; CHECK-LABEL: select_i8
; CHECK:       {{cmp w0, #0|tst w0, #0x1}}
; CHECK-NEXT:  csel {{w[0-9]+}}, w1, w2, ne
  %1 = select i1 %c, i8 %a, i8 %b
  ret i8 %1
}

define zeroext i16 @select_i16(i1 zeroext %c, i16 zeroext %a, i16 zeroext %b) {
; CHECK-LABEL: select_i16
; CHECK:       {{cmp w0, #0|tst w0, #0x1}}
; CHECK-NEXT:  csel {{w[0-9]+}}, w1, w2, ne
  %1 = select i1 %c, i16 %a, i16 %b
  ret i16 %1
}

define i32 @select_i32(i1 zeroext %c, i32 %a, i32 %b) {
; CHECK-LABEL: select_i32
; CHECK:       {{cmp w0, #0|tst w0, #0x1}}
; CHECK-NEXT:  csel {{w[0-9]+}}, w1, w2, ne
  %1 = select i1 %c, i32 %a, i32 %b
  ret i32 %1
}

define i64 @select_i64(i1 zeroext %c, i64 %a, i64 %b) {
; CHECK-LABEL: select_i64
; CHECK:       {{cmp w0, #0|tst w0, #0x1}}
; CHECK-NEXT:  csel {{x[0-9]+}}, x1, x2, ne
  %1 = select i1 %c, i64 %a, i64 %b
  ret i64 %1
}

define float @select_f32(i1 zeroext %c, float %a, float %b) {
; CHECK-LABEL: select_f32
; CHECK:       {{cmp w0, #0|tst w0, #0x1}}
; CHECK-NEXT:  fcsel {{s[0-9]+}}, s0, s1, ne
; GISEL-LABEL: select_f32
; GISEL:       {{cmp w0, #0|tst w0, #0x1}}
; GISEL-NEXT:  fcsel {{s[0-9]+}}, s0, s1, ne
  %1 = select i1 %c, float %a, float %b
  ret float %1
}

define double @select_f64(i1 zeroext %c, double %a, double %b) {
; CHECK-LABEL: select_f64
; CHECK:       {{cmp w0, #0|tst w0, #0x1}}
; CHECK-NEXT:  fcsel {{d[0-9]+}}, d0, d1, ne
; GISEL-LABEL: select_f64
; GISEL:       {{cmp w0, #0|tst w0, #0x1}}
; GISEL-NEXT:  fcsel {{d[0-9]+}}, d0, d1, ne
  %1 = select i1 %c, double %a, double %b
  ret double %1
}

; Now test the folding of all compares.
define float @select_fcmp_false(float %x, float %a, float %b) {
; CHECK-LABEL: select_fcmp_false
; CHECK:       mov.16b {{v[0-9]+}}, v2
  %1 = fcmp ogt float %x, %x
  %2 = select i1 %1, float %a, float %b
  ret float %2
}

define float @select_fcmp_ogt(float %x, float %y, float %a, float %b) {
; CHECK-LABEL: select_fcmp_ogt
; CHECK:       fcmp s0, s1
; CHECK-NEXT:  fcsel {{s[0-9]+}}, s2, s3, gt
  %1 = fcmp ogt float %x, %y
  %2 = select i1 %1, float %a, float %b
  ret float %2
}

define float @select_fcmp_oge(float %x, float %y, float %a, float %b) {
; CHECK-LABEL: select_fcmp_oge
; CHECK:       fcmp s0, s1
; CHECK-NEXT:  fcsel {{s[0-9]+}}, s2, s3, ge
  %1 = fcmp oge float %x, %y
  %2 = select i1 %1, float %a, float %b
  ret float %2
}

define float @select_fcmp_olt(float %x, float %y, float %a, float %b) {
; CHECK-LABEL: select_fcmp_olt
; CHECK:       fcmp s0, s1
; CHECK-NEXT:  fcsel {{s[0-9]+}}, s2, s3, mi
  %1 = fcmp olt float %x, %y
  %2 = select i1 %1, float %a, float %b
  ret float %2
}

define float @select_fcmp_ole(float %x, float %y, float %a, float %b) {
; CHECK-LABEL: select_fcmp_ole
; CHECK:       fcmp s0, s1
; CHECK-NEXT:  fcsel {{s[0-9]+}}, s2, s3, ls
  %1 = fcmp ole float %x, %y
  %2 = select i1 %1, float %a, float %b
  ret float %2
}

define float @select_fcmp_one(float %x, float %y, float %a, float %b) {
; CHECK-LABEL: select_fcmp_one
; CHECK:       fcmp s0, s1
; CHECK-NEXT:  fcsel [[REG:s[0-9]+]], s2, s3, mi
; CHECK-NEXT:  fcsel {{s[0-9]+}}, s2, [[REG]], gt
  %1 = fcmp one float %x, %y
  %2 = select i1 %1, float %a, float %b
  ret float %2
}

define float @select_fcmp_ord(float %x, float %y, float %a, float %b) {
; CHECK-LABEL: select_fcmp_ord
; CHECK:       fcmp s0, s1
; CHECK-NEXT:  fcsel {{s[0-9]+}}, s2, s3, vc
  %1 = fcmp ord float %x, %y
  %2 = select i1 %1, float %a, float %b
  ret float %2
}

define float @select_fcmp_uno(float %x, float %y, float %a, float %b) {
; CHECK-LABEL: select_fcmp_uno
; CHECK:       fcmp s0, s1
; CHECK-NEXT:  fcsel {{s[0-9]+}}, s2, s3, vs
  %1 = fcmp uno float %x, %y
  %2 = select i1 %1, float %a, float %b
  ret float %2
}

define float @select_fcmp_ueq(float %x, float %y, float %a, float %b) {
; CHECK-LABEL: select_fcmp_ueq
; CHECK:       fcmp s0, s1
; CHECK-NEXT:  fcsel [[REG:s[0-9]+]], s2, s3, eq
; CHECK-NEXT:  fcsel {{s[0-9]+}}, s2, [[REG]], vs
  %1 = fcmp ueq float %x, %y
  %2 = select i1 %1, float %a, float %b
  ret float %2
}

define float @select_fcmp_ugt(float %x, float %y, float %a, float %b) {
; CHECK-LABEL: select_fcmp_ugt
; CHECK:       fcmp s0, s1
; CHECK-NEXT:  fcsel {{s[0-9]+}}, s2, s3, hi
  %1 = fcmp ugt float %x, %y
  %2 = select i1 %1, float %a, float %b
  ret float %2
}

define float @select_fcmp_uge(float %x, float %y, float %a, float %b) {
; CHECK-LABEL: select_fcmp_uge
; CHECK:       fcmp s0, s1
; CHECK-NEXT:  fcsel {{s[0-9]+}}, s2, s3, pl
  %1 = fcmp uge float %x, %y
  %2 = select i1 %1, float %a, float %b
  ret float %2
}

define float @select_fcmp_ult(float %x, float %y, float %a, float %b) {
; CHECK-LABEL: select_fcmp_ult
; CHECK:       fcmp s0, s1
; CHECK-NEXT:  fcsel {{s[0-9]+}}, s2, s3, lt
  %1 = fcmp ult float %x, %y
  %2 = select i1 %1, float %a, float %b
  ret float %2
}


define float @select_fcmp_ule(float %x, float %y, float %a, float %b) {
; CHECK-LABEL: select_fcmp_ule
; CHECK:       fcmp s0, s1
; CHECK-NEXT:  fcsel {{s[0-9]+}}, s2, s3, le
  %1 = fcmp ule float %x, %y
  %2 = select i1 %1, float %a, float %b
  ret float %2
}

define float @select_fcmp_une(float %x, float %y, float %a, float %b) {
; CHECK-LABEL: select_fcmp_une
; CHECK:       fcmp s0, s1
; CHECK-NEXT:  fcsel {{s[0-9]+}}, s2, s3, ne
  %1 = fcmp une float %x, %y
  %2 = select i1 %1, float %a, float %b
  ret float %2
}

define float @select_fcmp_true(float %x, float %a, float %b) {
; CHECK-LABEL: select_fcmp_true
; CHECK:       mov.16b {{v[0-9]+}}, v1
  %1 = fcmp ueq float %x, %x
  %2 = select i1 %1, float %a, float %b
  ret float %2
}

define float @select_icmp_eq(i32 %x, i32 %y, float %a, float %b) {
; CHECK-LABEL: select_icmp_eq
; CHECK:       cmp w0, w1
; CHECK-NEXT:  fcsel {{s[0-9]+}}, s0, s1, eq
  %1 = icmp eq i32 %x, %y
  %2 = select i1 %1, float %a, float %b
  ret float %2
}

define float @select_icmp_ne(i32 %x, i32 %y, float %a, float %b) {
; CHECK-LABEL: select_icmp_ne
; CHECK:       cmp w0, w1
; CHECK-NEXT:  fcsel {{s[0-9]+}}, s0, s1, ne
  %1 = icmp ne i32 %x, %y
  %2 = select i1 %1, float %a, float %b
  ret float %2
}

define float @select_icmp_ugt(i32 %x, i32 %y, float %a, float %b) {
; CHECK-LABEL: select_icmp_ugt
; CHECK:       cmp w0, w1
; CHECK-NEXT:  fcsel {{s[0-9]+}}, s0, s1, hi
  %1 = icmp ugt i32 %x, %y
  %2 = select i1 %1, float %a, float %b
  ret float %2
}

define float @select_icmp_uge(i32 %x, i32 %y, float %a, float %b) {
; CHECK-LABEL: select_icmp_uge
; CHECK:       cmp w0, w1
; CHECK-NEXT:  fcsel {{s[0-9]+}}, s0, s1, hs
  %1 = icmp uge i32 %x, %y
  %2 = select i1 %1, float %a, float %b
  ret float %2
}

define float @select_icmp_ult(i32 %x, i32 %y, float %a, float %b) {
; CHECK-LABEL: select_icmp_ult
; CHECK:       cmp w0, w1
; CHECK-NEXT:  fcsel {{s[0-9]+}}, s0, s1, lo
  %1 = icmp ult i32 %x, %y
  %2 = select i1 %1, float %a, float %b
  ret float %2
}

define float @select_icmp_ule(i32 %x, i32 %y, float %a, float %b) {
; CHECK-LABEL: select_icmp_ule
; CHECK:       cmp w0, w1
; CHECK-NEXT:  fcsel {{s[0-9]+}}, s0, s1, ls
  %1 = icmp ule i32 %x, %y
  %2 = select i1 %1, float %a, float %b
  ret float %2
}

define float @select_icmp_sgt(i32 %x, i32 %y, float %a, float %b) {
; CHECK-LABEL: select_icmp_sgt
; CHECK:       cmp w0, w1
; CHECK-NEXT:  fcsel {{s[0-9]+}}, s0, s1, gt
  %1 = icmp sgt i32 %x, %y
  %2 = select i1 %1, float %a, float %b
  ret float %2
}

define float @select_icmp_sge(i32 %x, i32 %y, float %a, float %b) {
; CHECK-LABEL: select_icmp_sge
; CHECK:       cmp w0, w1
; CHECK-NEXT:  fcsel {{s[0-9]+}}, s0, s1, ge
  %1 = icmp sge i32 %x, %y
  %2 = select i1 %1, float %a, float %b
  ret float %2
}

define float @select_icmp_slt(i32 %x, i32 %y, float %a, float %b) {
; CHECK-LABEL: select_icmp_slt
; CHECK:       cmp w0, w1
; CHECK-NEXT:  fcsel {{s[0-9]+}}, s0, s1, lt
  %1 = icmp slt i32 %x, %y
  %2 = select i1 %1, float %a, float %b
  ret float %2
}

define float @select_icmp_sle(i32 %x, i32 %y, float %a, float %b) {
; CHECK-LABEL: select_icmp_sle
; CHECK:       cmp w0, w1
; CHECK-NEXT:  fcsel {{s[0-9]+}}, s0, s1, le
  %1 = icmp sle i32 %x, %y
  %2 = select i1 %1, float %a, float %b
  ret float %2
}

; Test peephole optimizations for select.
define zeroext i1 @select_opt1(i1 zeroext %c, i1 zeroext %a) {
; CHECK-LABEL: select_opt1
; CHECK:       orr {{w[0-9]+}}, w0, w1
  %1 = select i1 %c, i1 true, i1 %a
  ret i1 %1
}

define zeroext i1 @select_opt2(i1 zeroext %c, i1 zeroext %a) {
; CHECK-LABEL: select_opt2
; CHECK:       eor [[REG:w[0-9]+]], w0, #0x1
; CHECK:       orr {{w[0-9]+}}, [[REG]], w1
  %1 = select i1 %c, i1 %a, i1 true
  ret i1 %1
}

define zeroext i1 @select_opt3(i1 zeroext %c, i1 zeroext %a) {
; CHECK-LABEL: select_opt3
; CHECK:       bic {{w[0-9]+}}, w1, w0
  %1 = select i1 %c, i1 false, i1 %a
  ret i1 %1
}

define zeroext i1 @select_opt4(i1 zeroext %c, i1 zeroext %a) {
; CHECK-LABEL: select_opt4
; CHECK:       and {{w[0-9]+}}, w0, w1
  %1 = select i1 %c, i1 %a, i1 false
  ret i1 %1
}
