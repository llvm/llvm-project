; RUN: llc < %s -mtriple=ve-unknown-unknown | FileCheck %s

define signext i32 @i() {
; CHECK-LABEL: i:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, -2147483648
; CHECK-NEXT:    b.l.t (, %s10)
  ret i32 -2147483648
}

define zeroext i32 @ui() {
; CHECK-LABEL: ui:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, -2147483648
; CHECK-NEXT:    and %s0, %s0, (32)0
; CHECK-NEXT:    b.l.t (, %s10)
  ret i32 -2147483648
}

define i64 @ll() {
; CHECK-LABEL: ll:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, -2147483648
; CHECK-NEXT:    b.l.t (, %s10)
  ret i64 -2147483648
}

define i64 @ull() {
; CHECK-LABEL: ull:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, -2147483648
; CHECK-NEXT:    and %s0, %s0, (32)0
; CHECK-NEXT:    b.l.t (, %s10)
  ret i64 2147483648
}

define signext i8 @d2c(double %x) {
; CHECK-LABEL: d2c:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cvt.w.d.sx.rz %s0, %s0
; CHECK-NEXT:    adds.w.sx %s0, %s0, (0)1
; CHECK-NEXT:    b.l.t (, %s10)
  %r = fptosi double %x to i8
  ret i8 %r
}

define zeroext i8 @d2uc(double %x) {
; CHECK-LABEL: d2uc:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cvt.w.d.sx.rz %s0, %s0
; CHECK-NEXT:    adds.w.zx %s0, %s0, (0)1
; CHECK-NEXT:    b.l.t (, %s10)
  %r = fptoui double %x to i8
  ret i8 %r
}

define signext i16 @d2s(double %x) {
; CHECK-LABEL: d2s:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cvt.w.d.sx.rz %s0, %s0
; CHECK-NEXT:    adds.w.sx %s0, %s0, (0)1
; CHECK-NEXT:    b.l.t (, %s10)
  %r = fptosi double %x to i16
  ret i16 %r
}

define zeroext i16 @d2us(double %x) {
; CHECK-LABEL: d2us:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cvt.w.d.sx.rz %s0, %s0
; CHECK-NEXT:    adds.w.zx %s0, %s0, (0)1
; CHECK-NEXT:    b.l.t (, %s10)
  %r = fptoui double %x to i16
  ret i16 %r
}

define signext i32 @d2i(double %x) {
; CHECK-LABEL: d2i:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cvt.w.d.sx.rz %s0, %s0
; CHECK-NEXT:    adds.w.sx %s0, %s0, (0)1
; CHECK-NEXT:    b.l.t (, %s10)
  %r = fptosi double %x to i32
  ret i32 %r
}

define zeroext i32 @d2ui(double %x) {
; CHECK-LABEL: d2ui:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cvt.l.d.rz %s0, %s0
; CHECK-NEXT:    and %s0, %s0, (32)0
; CHECK-NEXT:    b.l.t (, %s10)
  %r = fptoui double %x to i32
  ret i32 %r
}

define i64 @d2ll(double %x) {
; CHECK-LABEL: d2ll:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cvt.l.d.rz %s0, %s0
; CHECK-NEXT:    b.l.t (, %s10)
  %r = fptosi double %x to i64
  ret i64 %r
}

define i64 @d2ull(double %x) {
; CHECK-LABEL: d2ull:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea.sl %s1, 1138753536
; CHECK-NEXT:    fcmp.d %s2, %s0, %s1
; CHECK-NEXT:    fsub.d %s1, %s0, %s1
; CHECK-NEXT:    cvt.l.d.rz %s1, %s1
; CHECK-NEXT:    xor %s1, %s1, (1)1
; CHECK-NEXT:    cvt.l.d.rz %s0, %s0
; CHECK-NEXT:    cmov.d.lt %s1, %s0, %s2
; CHECK-NEXT:    or %s0, 0, %s1
; CHECK-NEXT:    b.l.t (, %s10)
  %r = fptoui double %x to i64
  ret i64 %r
}

define float @d2f(double %x) {
; CHECK-LABEL: d2f:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cvt.s.d %s0, %s0
; CHECK-NEXT:    b.l.t (, %s10)
  %r = fptrunc double %x to float
  ret float %r
}

define double @d2d(double returned %0) {
; CHECK-LABEL: d2d:
; CHECK:       # %bb.0:
; CHECK-NEXT:    b.l.t (, %s10)
  ret double %0
}

define fp128 @d2q(double) {
; CHECK-LABEL: d2q:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cvt.q.d %s0, %s0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = fpext double %0 to fp128
  ret fp128 %2
}

define signext i8 @q2c(fp128) {
; CHECK-LABEL: q2c:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cvt.d.q %s0, %s0
; CHECK-NEXT:    cvt.w.d.sx.rz %s0, %s0
; CHECK-NEXT:    adds.w.sx %s0, %s0, (0)1
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = fptosi fp128 %0 to i8
  ret i8 %2
}

define zeroext i8 @q2uc(fp128) {
; CHECK-LABEL: q2uc:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cvt.d.q %s0, %s0
; CHECK-NEXT:    cvt.w.d.sx.rz %s0, %s0
; CHECK-NEXT:    adds.w.zx %s0, %s0, (0)1
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = fptoui fp128 %0 to i8
  ret i8 %2
}

define signext i16 @q2s(fp128) {
; CHECK-LABEL: q2s:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cvt.d.q %s0, %s0
; CHECK-NEXT:    cvt.w.d.sx.rz %s0, %s0
; CHECK-NEXT:    adds.w.sx %s0, %s0, (0)1
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = fptosi fp128 %0 to i16
  ret i16 %2
}

define zeroext i16 @q2us(fp128) {
; CHECK-LABEL: q2us:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cvt.d.q %s0, %s0
; CHECK-NEXT:    cvt.w.d.sx.rz %s0, %s0
; CHECK-NEXT:    adds.w.zx %s0, %s0, (0)1
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = fptoui fp128 %0 to i16
  ret i16 %2
}

define signext i32 @q2i(fp128) {
; CHECK-LABEL: q2i:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cvt.d.q %s0, %s0
; CHECK-NEXT:    cvt.w.d.sx.rz %s0, %s0
; CHECK-NEXT:    adds.w.sx %s0, %s0, (0)1
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = fptosi fp128 %0 to i32
  ret i32 %2
}

define zeroext i32 @q2ui(fp128) {
; CHECK-LABEL: q2ui:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cvt.d.q %s0, %s0
; CHECK-NEXT:    cvt.l.d.rz %s0, %s0
; CHECK-NEXT:    and %s0, %s0, (32)0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = fptoui fp128 %0 to i32
  ret i32 %2
}

define i64 @q2ll(fp128) {
; CHECK-LABEL: q2ll:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cvt.d.q %s0, %s0
; CHECK-NEXT:    cvt.l.d.rz %s0, %s0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = fptosi fp128 %0 to i64
  ret i64 %2
}

define i64 @q2ull(fp128) {
; CHECK-LABEL: q2ull:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s2, .LCPI{{[0-9]+}}_0@lo
; CHECK-NEXT:    and %s2, %s2, (32)0
; CHECK-NEXT:    lea.sl %s2, .LCPI{{[0-9]+}}_0@hi(, %s2)
; CHECK-NEXT:    ld %s4, 8(, %s2)
; CHECK-NEXT:    ld %s5, (, %s2)
; CHECK-NEXT:    fcmp.q %s3, %s0, %s4
; CHECK-NEXT:    fsub.q %s4, %s0, %s4
; CHECK-NEXT:    cvt.d.q %s2, %s4
; CHECK-NEXT:    cvt.l.d.rz %s2, %s2
; CHECK-NEXT:    xor %s2, %s2, (1)1
; CHECK-NEXT:    cvt.d.q %s0, %s0
; CHECK-NEXT:    cvt.l.d.rz %s0, %s0
; CHECK-NEXT:    cmov.d.lt %s2, %s0, %s3
; CHECK-NEXT:    or %s0, 0, %s2
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = fptoui fp128 %0 to i64
  ret i64 %2
}

define float @q2f(fp128) {
; CHECK-LABEL: q2f:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cvt.s.q %s0, %s0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = fptrunc fp128 %0 to float
  ret float %2
}

define double @q2d(fp128) {
; CHECK-LABEL: q2d:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cvt.d.q %s0, %s0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = fptrunc fp128 %0 to double
  ret double %2
}

define fp128 @q2q(fp128 returned) {
; CHECK-LABEL: q2q:
; CHECK:       # %bb.0:
; CHECK-NEXT:    b.l.t (, %s10)
  ret fp128 %0
}

define signext i8 @f2c(float %x) {
; CHECK-LABEL: f2c:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cvt.w.s.sx.rz %s0, %s0
; CHECK-NEXT:    adds.w.sx %s0, %s0, (0)1
; CHECK-NEXT:    b.l.t (, %s10)
  %r = fptosi float %x to i8
  ret i8 %r
}

define zeroext i8 @f2uc(float %x) {
; CHECK-LABEL: f2uc:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cvt.w.s.sx.rz %s0, %s0
; CHECK-NEXT:    adds.w.zx %s0, %s0, (0)1
; CHECK-NEXT:    b.l.t (, %s10)
  %r = fptoui float %x to i8
  ret i8 %r
}

define signext i16 @f2s(float %x) {
; CHECK-LABEL: f2s:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cvt.w.s.sx.rz %s0, %s0
; CHECK-NEXT:    adds.w.sx %s0, %s0, (0)1
; CHECK-NEXT:    b.l.t (, %s10)
  %r = fptosi float %x to i16
  ret i16 %r
}

define zeroext i16 @f2us(float %x) {
; CHECK-LABEL: f2us:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cvt.w.s.sx.rz %s0, %s0
; CHECK-NEXT:    adds.w.zx %s0, %s0, (0)1
; CHECK-NEXT:    b.l.t (, %s10)
  %r = fptoui float %x to i16
  ret i16 %r
}

define signext i32 @f2i(float %x) {
; CHECK-LABEL: f2i:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cvt.w.s.sx.rz %s0, %s0
; CHECK-NEXT:    adds.w.sx %s0, %s0, (0)1
; CHECK-NEXT:    b.l.t (, %s10)
  %r = fptosi float %x to i32
  ret i32 %r
}

define zeroext i32 @f2ui(float %x) {
; CHECK-LABEL: f2ui:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cvt.d.s %s0, %s0
; CHECK-NEXT:    cvt.l.d.rz %s0, %s0
; CHECK-NEXT:    and %s0, %s0, (32)0
; CHECK-NEXT:    b.l.t (, %s10)
  %r = fptoui float %x to i32
  ret i32 %r
}

define i64 @f2ll(float %x) {
; CHECK-LABEL: f2ll:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cvt.d.s %s0, %s0
; CHECK-NEXT:    cvt.l.d.rz %s0, %s0
; CHECK-NEXT:    b.l.t (, %s10)
  %r = fptosi float %x to i64
  ret i64 %r
}

define i64 @f2ull(float %x) {
; CHECK-LABEL: f2ull:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea.sl %s1, 1593835520
; CHECK-NEXT:    fcmp.s %s2, %s0, %s1
; CHECK-NEXT:    fsub.s %s1, %s0, %s1
; CHECK-NEXT:    cvt.d.s %s1, %s1
; CHECK-NEXT:    cvt.l.d.rz %s1, %s1
; CHECK-NEXT:    xor %s1, %s1, (1)1
; CHECK-NEXT:    cvt.d.s %s0, %s0
; CHECK-NEXT:    cvt.l.d.rz %s0, %s0
; CHECK-NEXT:    cmov.s.lt %s1, %s0, %s2
; CHECK-NEXT:    or %s0, 0, %s1
; CHECK-NEXT:    b.l.t (, %s10)
  %r = fptoui float %x to i64
  ret i64 %r
}

define float @f2f(float returned %0) {
; CHECK-LABEL: f2f:
; CHECK:       # %bb.0:
; CHECK-NEXT:    b.l.t (, %s10)
  ret float %0
}

define double @f2d(float %x) {
; CHECK-LABEL: f2d:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cvt.d.s %s0, %s0
; CHECK-NEXT:    b.l.t (, %s10)
  %r = fpext float %x to double
  ret double %r
}

define fp128 @f2q(float) {
; CHECK-LABEL: f2q:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cvt.q.s %s0, %s0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = fpext float %0 to fp128
  ret fp128 %2
}

define signext i8 @ll2c(i64 %0) {
; CHECK-LABEL: ll2c:
; CHECK:       # %bb.0:
; CHECK-NEXT:    sll %s0, %s0, 56
; CHECK-NEXT:    sra.l %s0, %s0, 56
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = trunc i64 %0 to i8
  ret i8 %2
}

define zeroext i8 @ll2uc(i64 %0) {
; CHECK-LABEL: ll2uc:
; CHECK:       # %bb.0:
; CHECK-NEXT:    and %s0, %s0, (56)0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = trunc i64 %0 to i8
  ret i8 %2
}

define signext i16 @ll2s(i64 %0) {
; CHECK-LABEL: ll2s:
; CHECK:       # %bb.0:
; CHECK-NEXT:    sll %s0, %s0, 48
; CHECK-NEXT:    sra.l %s0, %s0, 48
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = trunc i64 %0 to i16
  ret i16 %2
}

define zeroext i16 @ll2us(i64 %0) {
; CHECK-LABEL: ll2us:
; CHECK:       # %bb.0:
; CHECK-NEXT:    and %s0, %s0, (48)0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = trunc i64 %0 to i16
  ret i16 %2
}

define signext i32 @ll2i(i64 %0) {
; CHECK-LABEL: ll2i:
; CHECK:       # %bb.0:
; CHECK-NEXT:    adds.w.sx %s0, %s0, (0)1
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = trunc i64 %0 to i32
  ret i32 %2
}

define zeroext i32 @ll2ui(i64 %0) {
; CHECK-LABEL: ll2ui:
; CHECK:       # %bb.0:
; CHECK-NEXT:    and %s0, %s0, (32)0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = trunc i64 %0 to i32
  ret i32 %2
}

define i64 @ll2ll(i64 returned %0) {
; CHECK-LABEL: ll2ll:
; CHECK:       # %bb.0:
; CHECK-NEXT:    b.l.t (, %s10)
  ret i64 %0
}

define i64 @ll2ull(i64 returned %0) {
; CHECK-LABEL: ll2ull:
; CHECK:       # %bb.0:
; CHECK-NEXT:    b.l.t (, %s10)
  ret i64 %0
}

define float @ll2f(i64 %x) {
; CHECK-LABEL: ll2f:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cvt.d.l %s0, %s0
; CHECK-NEXT:    cvt.s.d %s0, %s0
; CHECK-NEXT:    b.l.t (, %s10)
  %r = sitofp i64 %x to float
  ret float %r
}

define double @ll2d(i64 %x) {
; CHECK-LABEL: ll2d:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cvt.d.l %s0, %s0
; CHECK-NEXT:    b.l.t (, %s10)
  %r = sitofp i64 %x to double
  ret double %r
}

define fp128 @ll2q(i64) {
; CHECK-LABEL: ll2q:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cvt.d.l %s0, %s0
; CHECK-NEXT:    cvt.q.d %s0, %s0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = sitofp i64 %0 to fp128
  ret fp128 %2
}

define signext i8 @ull2c(i64 %0) {
; CHECK-LABEL: ull2c:
; CHECK:       # %bb.0:
; CHECK-NEXT:    sll %s0, %s0, 56
; CHECK-NEXT:    sra.l %s0, %s0, 56
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = trunc i64 %0 to i8
  ret i8 %2
}

define zeroext i8 @ull2uc(i64 %0) {
; CHECK-LABEL: ull2uc:
; CHECK:       # %bb.0:
; CHECK-NEXT:    and %s0, %s0, (56)0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = trunc i64 %0 to i8
  ret i8 %2
}

define signext i16 @ull2s(i64 %0) {
; CHECK-LABEL: ull2s:
; CHECK:       # %bb.0:
; CHECK-NEXT:    sll %s0, %s0, 48
; CHECK-NEXT:    sra.l %s0, %s0, 48
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = trunc i64 %0 to i16
  ret i16 %2
}

define zeroext i16 @ull2us(i64 %0) {
; CHECK-LABEL: ull2us:
; CHECK:       # %bb.0:
; CHECK-NEXT:    and %s0, %s0, (48)0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = trunc i64 %0 to i16
  ret i16 %2
}

define signext i32 @ull2i(i64 %0) {
; CHECK-LABEL: ull2i:
; CHECK:       # %bb.0:
; CHECK-NEXT:    adds.w.sx %s0, %s0, (0)1
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = trunc i64 %0 to i32
  ret i32 %2
}

define zeroext i32 @ull2ui(i64 %0) {
; CHECK-LABEL: ull2ui:
; CHECK:       # %bb.0:
; CHECK-NEXT:    and %s0, %s0, (32)0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = trunc i64 %0 to i32
  ret i32 %2
}

define i64 @ull2ll(i64 returned %0) {
; CHECK-LABEL: ull2ll:
; CHECK:       # %bb.0:
; CHECK-NEXT:    b.l.t (, %s10)
  ret i64 %0
}

define i64 @ull2ull(i64 returned %0) {
; CHECK-LABEL: ull2ull:
; CHECK:       # %bb.0:
; CHECK-NEXT:    b.l.t (, %s10)
  ret i64 %0
}

define float @ull2f(i64 %x) {
; CHECK-LABEL: ull2f:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cmps.l %s2, %s0, (0)1
; CHECK-NEXT:    cvt.d.l %s1, %s0
; CHECK-NEXT:    cvt.s.d %s1, %s1
; CHECK-NEXT:    srl %s3, %s0, 1
; CHECK-NEXT:    and %s0, 1, %s0
; CHECK-NEXT:    or %s0, %s0, %s3
; CHECK-NEXT:    cvt.d.l %s0, %s0
; CHECK-NEXT:    cvt.s.d %s0, %s0
; CHECK-NEXT:    fadd.s %s0, %s0, %s0
; CHECK-NEXT:    cmov.l.lt %s1, %s0, %s2
; CHECK-NEXT:    or %s0, 0, %s1
; CHECK-NEXT:    b.l.t (, %s10)
  %r = uitofp i64 %x to float
  ret float %r
}

define double @ull2d(i64 %x) {
; CHECK-LABEL: ull2d:
; CHECK:       # %bb.0:
; CHECK-NEXT:    srl %s1, %s0, 32
; CHECK-NEXT:    lea.sl %s2, 1160773632
; CHECK-NEXT:    or %s1, %s1, %s2
; CHECK-NEXT:    lea %s2, 1048576
; CHECK-NEXT:    lea.sl %s2, -986710016(, %s2)
; CHECK-NEXT:    fadd.d %s1, %s1, %s2
; CHECK-NEXT:    and %s0, %s0, (32)0
; CHECK-NEXT:    lea.sl %s2, 1127219200
; CHECK-NEXT:    or %s0, %s0, %s2
; CHECK-NEXT:    fadd.d %s0, %s0, %s1
; CHECK-NEXT:    b.l.t (, %s10)
  %r = uitofp i64 %x to double
  ret double %r
}

define fp128 @ull2q(i64) {
; CHECK-LABEL: ull2q:
; CHECK:       # %bb.0:
; CHECK-NEXT:    srl %s1, %s0, 61
; CHECK-NEXT:    and %s1, 4, %s1
; CHECK-NEXT:    lea %s2, .LCPI{{[0-9]+}}_0@lo
; CHECK-NEXT:    and %s2, %s2, (32)0
; CHECK-NEXT:    lea.sl %s2, .LCPI{{[0-9]+}}_0@hi(, %s2)
; CHECK-NEXT:    ldu %s1, (%s1, %s2)
; CHECK-NEXT:    cvt.q.s %s2, %s1
; CHECK-NEXT:    cvt.d.l %s0, %s0
; CHECK-NEXT:    cvt.q.d %s0, %s0
; CHECK-NEXT:    fadd.q %s0, %s0, %s2
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = uitofp i64 %0 to fp128
  ret fp128 %2
}

define signext i8 @i2c(i32 signext %0) {
; CHECK-LABEL: i2c:
; CHECK:       # %bb.0:
; CHECK-NEXT:    sll %s0, %s0, 56
; CHECK-NEXT:    sra.l %s0, %s0, 56
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = trunc i32 %0 to i8
  ret i8 %2
}

define zeroext i8 @i2uc(i32 signext %0) {
; CHECK-LABEL: i2uc:
; CHECK:       # %bb.0:
; CHECK-NEXT:    and %s0, %s0, (56)0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = trunc i32 %0 to i8
  ret i8 %2
}

define signext i16 @i2s(i32 signext %0) {
; CHECK-LABEL: i2s:
; CHECK:       # %bb.0:
; CHECK-NEXT:    sll %s0, %s0, 48
; CHECK-NEXT:    sra.l %s0, %s0, 48
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = trunc i32 %0 to i16
  ret i16 %2
}

define zeroext i16 @i2us(i32 signext %0) {
; CHECK-LABEL: i2us:
; CHECK:       # %bb.0:
; CHECK-NEXT:    and %s0, %s0, (48)0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = trunc i32 %0 to i16
  ret i16 %2
}

define signext i32 @i2i(i32 signext returned %0) {
; CHECK-LABEL: i2i:
; CHECK:       # %bb.0:
; CHECK-NEXT:    b.l.t (, %s10)
  ret i32 %0
}

define zeroext i32 @i2ui(i32 signext returned %0) {
; CHECK-LABEL: i2ui:
; CHECK:       # %bb.0:
; CHECK-NEXT:    and %s0, %s0, (32)0
; CHECK-NEXT:    b.l.t (, %s10)
  ret i32 %0
}

define i64 @i2ll(i32 signext %0) {
; CHECK-LABEL: i2ll:
; CHECK:       # %bb.0:
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = sext i32 %0 to i64
  ret i64 %2
}

define i64 @i2ull(i32 signext %0) {
; CHECK-LABEL: i2ull:
; CHECK:       # %bb.0:
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = sext i32 %0 to i64
  ret i64 %2
}

define float @i2f(i32 signext %x) {
; CHECK-LABEL: i2f:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cvt.s.w %s0, %s0
; CHECK-NEXT:    b.l.t (, %s10)
  %r = sitofp i32 %x to float
  ret float %r
}

define double @i2d(i32 signext %x) {
; CHECK-LABEL: i2d:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cvt.d.w %s0, %s0
; CHECK-NEXT:    b.l.t (, %s10)
  %r = sitofp i32 %x to double
  ret double %r
}

define fp128 @i2q(i32 signext %x) {
; CHECK-LABEL: i2q:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cvt.d.w %s0, %s0
; CHECK-NEXT:    cvt.q.d %s0, %s0
; CHECK-NEXT:    b.l.t (, %s10)
  %r = sitofp i32 %x to fp128
  ret fp128 %r
}

define signext i8 @ui2c(i32 zeroext %0) {
; CHECK-LABEL: ui2c:
; CHECK:       # %bb.0:
; CHECK-NEXT:    sll %s0, %s0, 56
; CHECK-NEXT:    sra.l %s0, %s0, 56
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = trunc i32 %0 to i8
  ret i8 %2
}

define zeroext i8 @ui2uc(i32 zeroext %0) {
; CHECK-LABEL: ui2uc:
; CHECK:       # %bb.0:
; CHECK-NEXT:    and %s0, %s0, (56)0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = trunc i32 %0 to i8
  ret i8 %2
}

define signext i16 @ui2s(i32 zeroext %0) {
; CHECK-LABEL: ui2s:
; CHECK:       # %bb.0:
; CHECK-NEXT:    sll %s0, %s0, 48
; CHECK-NEXT:    sra.l %s0, %s0, 48
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = trunc i32 %0 to i16
  ret i16 %2
}

define zeroext i16 @ui2us(i32 zeroext %0) {
; CHECK-LABEL: ui2us:
; CHECK:       # %bb.0:
; CHECK-NEXT:    and %s0, %s0, (48)0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = trunc i32 %0 to i16
  ret i16 %2
}

define signext i32 @ui2i(i32 zeroext returned %0) {
; CHECK-LABEL: ui2i:
; CHECK:       # %bb.0:
; CHECK-NEXT:    adds.w.sx %s0, %s0, (0)1
; CHECK-NEXT:    b.l.t (, %s10)
  ret i32 %0
}

define zeroext i32 @ui2ui(i32 zeroext returned %0) {
; CHECK-LABEL: ui2ui:
; CHECK:       # %bb.0:
; CHECK-NEXT:    b.l.t (, %s10)
  ret i32 %0
}

define i64 @ui2ll(i32 zeroext %0) {
; CHECK-LABEL: ui2ll:
; CHECK:       # %bb.0:
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = zext i32 %0 to i64
  ret i64 %2
}

define i64 @ui2ull(i32 zeroext %0) {
; CHECK-LABEL: ui2ull:
; CHECK:       # %bb.0:
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = zext i32 %0 to i64
  ret i64 %2
}

define float @ui2f(i32 zeroext %x) {
; CHECK-LABEL: ui2f:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cvt.d.l %s0, %s0
; CHECK-NEXT:    cvt.s.d %s0, %s0
; CHECK-NEXT:    b.l.t (, %s10)
  %r = uitofp i32 %x to float
  ret float %r
}

define double @ui2d(i32 zeroext %x) {
; CHECK-LABEL: ui2d:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cvt.d.l %s0, %s0
; CHECK-NEXT:    b.l.t (, %s10)
  %r = uitofp i32 %x to double
  ret double %r
}

define fp128 @ui2q(i32 zeroext %0) {
; CHECK-LABEL: ui2q:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cvt.d.l %s0, %s0
; CHECK-NEXT:    cvt.q.d %s0, %s0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = uitofp i32 %0 to fp128
  ret fp128 %2
}

define signext i8 @s2c(i16 signext %0) {
; CHECK-LABEL: s2c:
; CHECK:       # %bb.0:
; CHECK-NEXT:    sll %s0, %s0, 56
; CHECK-NEXT:    sra.l %s0, %s0, 56
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = trunc i16 %0 to i8
  ret i8 %2
}

define zeroext i8 @s2uc(i16 signext %0) {
; CHECK-LABEL: s2uc:
; CHECK:       # %bb.0:
; CHECK-NEXT:    and %s0, %s0, (56)0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = trunc i16 %0 to i8
  ret i8 %2
}

define signext i16 @s2s(i16 returned signext %0) {
; CHECK-LABEL: s2s:
; CHECK:       # %bb.0:
; CHECK-NEXT:    b.l.t (, %s10)
  ret i16 %0
}

define zeroext i16 @s2us(i16 returned signext %0) {
; CHECK-LABEL: s2us:
; CHECK:       # %bb.0:
; CHECK-NEXT:    and %s0, %s0, (48)0
; CHECK-NEXT:    b.l.t (, %s10)
  ret i16 %0
}

define signext i32 @s2i(i16 signext %0) {
; CHECK-LABEL: s2i:
; CHECK:       # %bb.0:
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = sext i16 %0 to i32
  ret i32 %2
}

define zeroext i32 @s2ui(i16 signext %0) {
; CHECK-LABEL: s2ui:
; CHECK:       # %bb.0:
; CHECK-NEXT:    and %s0, %s0, (32)0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = sext i16 %0 to i32
  ret i32 %2
}

define i64 @s2ll(i16 signext %0) {
; CHECK-LABEL: s2ll:
; CHECK:       # %bb.0:
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = sext i16 %0 to i64
  ret i64 %2
}

define i64 @s2ull(i16 signext %0) {
; CHECK-LABEL: s2ull:
; CHECK:       # %bb.0:
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = sext i16 %0 to i64
  ret i64 %2
}

define float @s2f(i16 signext %x) {
; CHECK-LABEL: s2f:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cvt.s.w %s0, %s0
; CHECK-NEXT:    b.l.t (, %s10)
  %r = sitofp i16 %x to float
  ret float %r
}

define double @s2d(i16 signext %x) {
; CHECK-LABEL: s2d:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cvt.d.w %s0, %s0
; CHECK-NEXT:    b.l.t (, %s10)
  %r = sitofp i16 %x to double
  ret double %r
}

define fp128 @s2q(i16 signext) {
; CHECK-LABEL: s2q:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cvt.d.w %s0, %s0
; CHECK-NEXT:    cvt.q.d %s0, %s0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = sitofp i16 %0 to fp128
  ret fp128 %2
}

define signext i8 @us2c(i16 zeroext %0) {
; CHECK-LABEL: us2c:
; CHECK:       # %bb.0:
; CHECK-NEXT:    sll %s0, %s0, 56
; CHECK-NEXT:    sra.l %s0, %s0, 56
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = trunc i16 %0 to i8
  ret i8 %2
}

define zeroext i8 @us2uc(i16 zeroext %0) {
; CHECK-LABEL: us2uc:
; CHECK:       # %bb.0:
; CHECK-NEXT:    and %s0, %s0, (56)0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = trunc i16 %0 to i8
  ret i8 %2
}

define signext i16 @us2s(i16 returned zeroext %0) {
; CHECK-LABEL: us2s:
; CHECK:       # %bb.0:
; CHECK-NEXT:    sll %s0, %s0, 48
; CHECK-NEXT:    sra.l %s0, %s0, 48
; CHECK-NEXT:    b.l.t (, %s10)
  ret i16 %0
}

define zeroext i16 @us2us(i16 returned zeroext %0) {
; CHECK-LABEL: us2us:
; CHECK:       # %bb.0:
; CHECK-NEXT:    b.l.t (, %s10)
  ret i16 %0
}

define signext i32 @us2i(i16 zeroext %0) {
; CHECK-LABEL: us2i:
; CHECK:       # %bb.0:
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = zext i16 %0 to i32
  ret i32 %2
}

define zeroext i32 @us2ui(i16 zeroext %0) {
; CHECK-LABEL: us2ui:
; CHECK:       # %bb.0:
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = zext i16 %0 to i32
  ret i32 %2
}

define i64 @us2ll(i16 zeroext %0) {
; CHECK-LABEL: us2ll:
; CHECK:       # %bb.0:
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = zext i16 %0 to i64
  ret i64 %2
}

define i64 @us2ull(i16 zeroext %0) {
; CHECK-LABEL: us2ull:
; CHECK:       # %bb.0:
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = zext i16 %0 to i64
  ret i64 %2
}

define float @us2f(i16 zeroext %x) {
; CHECK-LABEL: us2f:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cvt.s.w %s0, %s0
; CHECK-NEXT:    b.l.t (, %s10)
  %r = uitofp i16 %x to float
  ret float %r
}

define double @us2d(i16 zeroext %x) {
; CHECK-LABEL: us2d:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cvt.d.w %s0, %s0
; CHECK-NEXT:    b.l.t (, %s10)
  %r = uitofp i16 %x to double
  ret double %r
}

define fp128 @us2q(i16 zeroext) {
; CHECK-LABEL: us2q:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cvt.d.w %s0, %s0
; CHECK-NEXT:    cvt.q.d %s0, %s0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = uitofp i16 %0 to fp128
  ret fp128 %2
}

define signext i8 @c2c(i8 returned signext %0) {
; CHECK-LABEL: c2c:
; CHECK:       # %bb.0:
; CHECK-NEXT:    b.l.t (, %s10)
  ret i8 %0
}

define zeroext i8 @c2uc(i8 returned signext %0) {
; CHECK-LABEL: c2uc:
; CHECK:       # %bb.0:
; CHECK-NEXT:    and %s0, %s0, (56)0
; CHECK-NEXT:    b.l.t (, %s10)
  ret i8 %0
}

define signext i16 @c2s(i8 signext %0) {
; CHECK-LABEL: c2s:
; CHECK:       # %bb.0:
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = sext i8 %0 to i16
  ret i16 %2
}

define zeroext i16 @c2us(i8 signext %0) {
; CHECK-LABEL: c2us:
; CHECK:       # %bb.0:
; CHECK-NEXT:    and %s0, %s0, (48)0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = sext i8 %0 to i16
  ret i16 %2
}

define signext i32 @c2i(i8 signext %0) {
; CHECK-LABEL: c2i:
; CHECK:       # %bb.0:
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = sext i8 %0 to i32
  ret i32 %2
}

define zeroext i32 @c2ui(i8 signext %0) {
; CHECK-LABEL: c2ui:
; CHECK:       # %bb.0:
; CHECK-NEXT:    and %s0, %s0, (32)0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = sext i8 %0 to i32
  ret i32 %2
}

define i64 @c2ll(i8 signext %0) {
; CHECK-LABEL: c2ll:
; CHECK:       # %bb.0:
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = sext i8 %0 to i64
  ret i64 %2
}

define i64 @c2ull(i8 signext %0) {
; CHECK-LABEL: c2ull:
; CHECK:       # %bb.0:
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = sext i8 %0 to i64
  ret i64 %2
}

define float @c2f(i8 signext %x) {
; CHECK-LABEL: c2f:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cvt.s.w %s0, %s0
; CHECK-NEXT:    b.l.t (, %s10)
  %r = sitofp i8 %x to float
  ret float %r
}

define double @c2d(i8 signext %x) {
; CHECK-LABEL: c2d:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cvt.d.w %s0, %s0
; CHECK-NEXT:    b.l.t (, %s10)
  %r = sitofp i8 %x to double
  ret double %r
}

define fp128 @c2q(i8 signext) {
; CHECK-LABEL: c2q:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cvt.d.w %s0, %s0
; CHECK-NEXT:    cvt.q.d %s0, %s0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = sitofp i8 %0 to fp128
  ret fp128 %2
}

define signext i8 @uc2c(i8 returned zeroext %0) {
; CHECK-LABEL: uc2c:
; CHECK:       # %bb.0:
; CHECK-NEXT:    sll %s0, %s0, 56
; CHECK-NEXT:    sra.l %s0, %s0, 56
; CHECK-NEXT:    b.l.t (, %s10)
  ret i8 %0
}

define zeroext i8 @uc2uc(i8 returned zeroext %0) {
; CHECK-LABEL: uc2uc:
; CHECK:       # %bb.0:
; CHECK-NEXT:    b.l.t (, %s10)
  ret i8 %0
}

define signext i16 @uc2s(i8 zeroext %0) {
; CHECK-LABEL: uc2s:
; CHECK:       # %bb.0:
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = zext i8 %0 to i16
  ret i16 %2
}

define zeroext i16 @uc2us(i8 zeroext %0) {
; CHECK-LABEL: uc2us:
; CHECK:       # %bb.0:
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = zext i8 %0 to i16
  ret i16 %2
}

define signext i32 @uc2i(i8 zeroext %0) {
; CHECK-LABEL: uc2i:
; CHECK:       # %bb.0:
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = zext i8 %0 to i32
  ret i32 %2
}

define zeroext i32 @uc2ui(i8 zeroext %0) {
; CHECK-LABEL: uc2ui:
; CHECK:       # %bb.0:
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = zext i8 %0 to i32
  ret i32 %2
}

define i64 @uc2ll(i8 zeroext %0) {
; CHECK-LABEL: uc2ll:
; CHECK:       # %bb.0:
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = zext i8 %0 to i64
  ret i64 %2
}

define i64 @uc2ull(i8 zeroext %0) {
; CHECK-LABEL: uc2ull:
; CHECK:       # %bb.0:
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = zext i8 %0 to i64
  ret i64 %2
}

define float @uc2f(i8 zeroext %x) {
; CHECK-LABEL: uc2f:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cvt.s.w %s0, %s0
; CHECK-NEXT:    b.l.t (, %s10)
  %r = uitofp i8 %x to float
  ret float %r
}

define double @uc2d(i8 zeroext %x) {
; CHECK-LABEL: uc2d:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cvt.d.w %s0, %s0
; CHECK-NEXT:    b.l.t (, %s10)
  %r = uitofp i8 %x to double
  ret double %r
}

define fp128 @uc2q(i8 zeroext) {
; CHECK-LABEL: uc2q:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cvt.d.w %s0, %s0
; CHECK-NEXT:    cvt.q.d %s0, %s0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = uitofp i8 %0 to fp128
  ret fp128 %2
}

; Function Attrs: norecurse nounwind readnone
define i128 @i128() {
; CHECK-LABEL: i128:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, -2147483648
; CHECK-NEXT:    or %s1, -1, (0)1
; CHECK-NEXT:    b.l.t (, %s10)
  ret i128 -2147483648
}

; Function Attrs: norecurse nounwind readnone
define i128 @ui128() {
; CHECK-LABEL: ui128:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, -2147483648
; CHECK-NEXT:    or %s1, -1, (0)1
; CHECK-NEXT:    b.l.t (, %s10)
  ret i128 -2147483648
}

; Function Attrs: norecurse nounwind readnone
define signext i8 @i1282c(i128 %0) {
; CHECK-LABEL: i1282c:
; CHECK:       # %bb.0:
; CHECK-NEXT:    sll %s0, %s0, 56
; CHECK-NEXT:    sra.l %s0, %s0, 56
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = trunc i128 %0 to i8
  ret i8 %2
}

; Function Attrs: norecurse nounwind readnone
define signext i8 @ui1282c(i128 %0) {
; CHECK-LABEL: ui1282c:
; CHECK:       # %bb.0:
; CHECK-NEXT:    sll %s0, %s0, 56
; CHECK-NEXT:    sra.l %s0, %s0, 56
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = trunc i128 %0 to i8
  ret i8 %2
}

; Function Attrs: norecurse nounwind readnone
define zeroext i8 @i1282uc(i128 %0) {
; CHECK-LABEL: i1282uc:
; CHECK:       # %bb.0:
; CHECK-NEXT:    and %s0, %s0, (56)0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = trunc i128 %0 to i8
  ret i8 %2
}

; Function Attrs: norecurse nounwind readnone
define zeroext i8 @ui1282uc(i128 %0) {
; CHECK-LABEL: ui1282uc:
; CHECK:       # %bb.0:
; CHECK-NEXT:    and %s0, %s0, (56)0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = trunc i128 %0 to i8
  ret i8 %2
}

; Function Attrs: norecurse nounwind readnone
define signext i16 @i1282s(i128 %0) {
; CHECK-LABEL: i1282s:
; CHECK:       # %bb.0:
; CHECK-NEXT:    sll %s0, %s0, 48
; CHECK-NEXT:    sra.l %s0, %s0, 48
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = trunc i128 %0 to i16
  ret i16 %2
}

; Function Attrs: norecurse nounwind readnone
define signext i16 @ui1282s(i128 %0) {
; CHECK-LABEL: ui1282s:
; CHECK:       # %bb.0:
; CHECK-NEXT:    sll %s0, %s0, 48
; CHECK-NEXT:    sra.l %s0, %s0, 48
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = trunc i128 %0 to i16
  ret i16 %2
}

; Function Attrs: norecurse nounwind readnone
define zeroext i16 @i1282us(i128 %0) {
; CHECK-LABEL: i1282us:
; CHECK:       # %bb.0:
; CHECK-NEXT:    and %s0, %s0, (48)0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = trunc i128 %0 to i16
  ret i16 %2
}

; Function Attrs: norecurse nounwind readnone
define zeroext i16 @ui1282us(i128 %0) {
; CHECK-LABEL: ui1282us:
; CHECK:       # %bb.0:
; CHECK-NEXT:    and %s0, %s0, (48)0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = trunc i128 %0 to i16
  ret i16 %2
}

; Function Attrs: norecurse nounwind readnone
define signext i32 @i1282i(i128 %0) {
; CHECK-LABEL: i1282i:
; CHECK:       # %bb.0:
; CHECK-NEXT:    adds.w.sx %s0, %s0, (0)1
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = trunc i128 %0 to i32
  ret i32 %2
}

; Function Attrs: norecurse nounwind readnone
define signext i32 @ui1282i(i128 %0) {
; CHECK-LABEL: ui1282i:
; CHECK:       # %bb.0:
; CHECK-NEXT:    adds.w.sx %s0, %s0, (0)1
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = trunc i128 %0 to i32
  ret i32 %2
}

; Function Attrs: norecurse nounwind readnone
define zeroext i32 @i1282ui(i128 %0) {
; CHECK-LABEL: i1282ui:
; CHECK:       # %bb.0:
; CHECK-NEXT:    and %s0, %s0, (32)0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = trunc i128 %0 to i32
  ret i32 %2
}

; Function Attrs: norecurse nounwind readnone
define zeroext i32 @ui1282ui(i128 %0) {
; CHECK-LABEL: ui1282ui:
; CHECK:       # %bb.0:
; CHECK-NEXT:    and %s0, %s0, (32)0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = trunc i128 %0 to i32
  ret i32 %2
}

; Function Attrs: norecurse nounwind readnone
define i64 @i1282ll(i128 %0) {
; CHECK-LABEL: i1282ll:
; CHECK:       # %bb.0:
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = trunc i128 %0 to i64
  ret i64 %2
}

; Function Attrs: norecurse nounwind readnone
define i64 @ui1282ll(i128 %0) {
; CHECK-LABEL: ui1282ll:
; CHECK:       # %bb.0:
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = trunc i128 %0 to i64
  ret i64 %2
}

; Function Attrs: norecurse nounwind readnone
define i64 @i1282ull(i128 %0) {
; CHECK-LABEL: i1282ull:
; CHECK:       # %bb.0:
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = trunc i128 %0 to i64
  ret i64 %2
}

; Function Attrs: norecurse nounwind readnone
define i64 @ui1282ull(i128 %0) {
; CHECK-LABEL: ui1282ull:
; CHECK:       # %bb.0:
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = trunc i128 %0 to i64
  ret i64 %2
}

; Function Attrs: norecurse nounwind readnone
define i128 @i1282ui128(i128 returned %0) {
; CHECK-LABEL: i1282ui128:
; CHECK:       # %bb.0:
; CHECK-NEXT:    b.l.t (, %s10)
  ret i128 %0
}

; Function Attrs: norecurse nounwind readnone
define i128 @ui1282i128(i128 returned %0) {
; CHECK-LABEL: ui1282i128:
; CHECK:       # %bb.0:
; CHECK-NEXT:    b.l.t (, %s10)
  ret i128 %0
}

; Function Attrs: norecurse nounwind readnone
define float @i1282f(i128) {
; CHECK-LABEL: i1282f:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    lea %s2, __floattisf@lo
; CHECK-NEXT:    and %s2, %s2, (32)0
; CHECK-NEXT:    lea.sl %s12, __floattisf@hi(, %s2)
; CHECK-NEXT:    bsic %s10, (, %s12)
; CHECK-NEXT:    or %s11, 0, %s9
  %2 = sitofp i128 %0 to float
  ret float %2
}

; Function Attrs: norecurse nounwind readnone
define float @ui1282f(i128) {
; CHECK-LABEL: ui1282f:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    lea %s2, __floatuntisf@lo
; CHECK-NEXT:    and %s2, %s2, (32)0
; CHECK-NEXT:    lea.sl %s12, __floatuntisf@hi(, %s2)
; CHECK-NEXT:    bsic %s10, (, %s12)
; CHECK-NEXT:    or %s11, 0, %s9
  %2 = uitofp i128 %0 to float
  ret float %2
}

; Function Attrs: norecurse nounwind readnone
define double @i1282d(i128) {
; CHECK-LABEL: i1282d:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    lea %s2, __floattidf@lo
; CHECK-NEXT:    and %s2, %s2, (32)0
; CHECK-NEXT:    lea.sl %s12, __floattidf@hi(, %s2)
; CHECK-NEXT:    bsic %s10, (, %s12)
; CHECK-NEXT:    or %s11, 0, %s9
  %2 = sitofp i128 %0 to double
  ret double %2
}

; Function Attrs: norecurse nounwind readnone
define double @ui1282d(i128) {
; CHECK-LABEL: ui1282d:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    lea %s2, __floatuntidf@lo
; CHECK-NEXT:    and %s2, %s2, (32)0
; CHECK-NEXT:    lea.sl %s12, __floatuntidf@hi(, %s2)
; CHECK-NEXT:    bsic %s10, (, %s12)
; CHECK-NEXT:    or %s11, 0, %s9
  %2 = uitofp i128 %0 to double
  ret double %2
}

; Function Attrs: norecurse nounwind readnone
define i128 @d2i128(double) {
; CHECK-LABEL: d2i128:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    lea %s1, __fixdfti@lo
; CHECK-NEXT:    and %s1, %s1, (32)0
; CHECK-NEXT:    lea.sl %s12, __fixdfti@hi(, %s1)
; CHECK-NEXT:    bsic %s10, (, %s12)
; CHECK-NEXT:    or %s11, 0, %s9
  %2 = fptosi double %0 to i128
  ret i128 %2
}

; Function Attrs: norecurse nounwind readnone
define i128 @d2ui128(double) {
; CHECK-LABEL: d2ui128:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    lea %s1, __fixunsdfti@lo
; CHECK-NEXT:    and %s1, %s1, (32)0
; CHECK-NEXT:    lea.sl %s12, __fixunsdfti@hi(, %s1)
; CHECK-NEXT:    bsic %s10, (, %s12)
; CHECK-NEXT:    or %s11, 0, %s9
  %2 = fptoui double %0 to i128
  ret i128 %2
}

; Function Attrs: norecurse nounwind readnone
define i128 @f2i128(float) {
; CHECK-LABEL: f2i128:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    lea %s1, __fixsfti@lo
; CHECK-NEXT:    and %s1, %s1, (32)0
; CHECK-NEXT:    lea.sl %s12, __fixsfti@hi(, %s1)
; CHECK-NEXT:    bsic %s10, (, %s12)
; CHECK-NEXT:    or %s11, 0, %s9
  %2 = fptosi float %0 to i128
  ret i128 %2
}

; Function Attrs: norecurse nounwind readnone
define i128 @f2ui128(float) {
; CHECK-LABEL: f2ui128:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    lea %s1, __fixunssfti@lo
; CHECK-NEXT:    and %s1, %s1, (32)0
; CHECK-NEXT:    lea.sl %s12, __fixunssfti@hi(, %s1)
; CHECK-NEXT:    bsic %s10, (, %s12)
; CHECK-NEXT:    or %s11, 0, %s9
  %2 = fptoui float %0 to i128
  ret i128 %2
}

; Function Attrs: norecurse nounwind readnone
define i128 @ll2i128(i64 %0) {
; CHECK-LABEL: ll2i128:
; CHECK:       # %bb.0:
; CHECK-NEXT:    sra.l %s1, %s0, 63
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = sext i64 %0 to i128
  ret i128 %2
}

; Function Attrs: norecurse nounwind readnone
define i128 @ll2ui128(i64 %0) {
; CHECK-LABEL: ll2ui128:
; CHECK:       # %bb.0:
; CHECK-NEXT:    sra.l %s1, %s0, 63
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = sext i64 %0 to i128
  ret i128 %2
}

; Function Attrs: norecurse nounwind readnone
define i128 @ull2i128(i64 %0) {
; CHECK-LABEL: ull2i128:
; CHECK:       # %bb.0:
; CHECK-NEXT:    or %s1, 0, (0)1
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = zext i64 %0 to i128
  ret i128 %2
}

; Function Attrs: norecurse nounwind readnone
define i128 @ull2ui128(i64 %0) {
; CHECK-LABEL: ull2ui128:
; CHECK:       # %bb.0:
; CHECK-NEXT:    or %s1, 0, (0)1
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = zext i64 %0 to i128
  ret i128 %2
}

; Function Attrs: norecurse nounwind readnone
define i128 @i2i128(i32 signext %0) {
; CHECK-LABEL: i2i128:
; CHECK:       # %bb.0:
; CHECK-NEXT:    sra.l %s1, %s0, 63
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = sext i32 %0 to i128
  ret i128 %2
}

; Function Attrs: norecurse nounwind readnone
define i128 @i2ui128(i32 signext %0) {
; CHECK-LABEL: i2ui128:
; CHECK:       # %bb.0:
; CHECK-NEXT:    sra.l %s1, %s0, 63
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = sext i32 %0 to i128
  ret i128 %2
}

; Function Attrs: norecurse nounwind readnone
define i128 @ui2i128(i32 zeroext %0) {
; CHECK-LABEL: ui2i128:
; CHECK:       # %bb.0:
; CHECK-NEXT:    or %s1, 0, (0)1
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = zext i32 %0 to i128
  ret i128 %2
}

; Function Attrs: norecurse nounwind readnone
define i128 @ui2ui128(i32 zeroext %0) {
; CHECK-LABEL: ui2ui128:
; CHECK:       # %bb.0:
; CHECK-NEXT:    or %s1, 0, (0)1
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = zext i32 %0 to i128
  ret i128 %2
}

; Function Attrs: norecurse nounwind readnone
define i128 @s2i128(i16 signext %0) {
; CHECK-LABEL: s2i128:
; CHECK:       # %bb.0:
; CHECK-NEXT:    sra.l %s1, %s0, 63
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = sext i16 %0 to i128
  ret i128 %2
}

; Function Attrs: norecurse nounwind readnone
define i128 @s2ui128(i16 signext %0) {
; CHECK-LABEL: s2ui128:
; CHECK:       # %bb.0:
; CHECK-NEXT:    sra.l %s1, %s0, 63
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = sext i16 %0 to i128
  ret i128 %2
}

; Function Attrs: norecurse nounwind readnone
define i128 @us2i128(i16 zeroext %0) {
; CHECK-LABEL: us2i128:
; CHECK:       # %bb.0:
; CHECK-NEXT:    or %s1, 0, (0)1
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = zext i16 %0 to i128
  ret i128 %2
}

; Function Attrs: norecurse nounwind readnone
define i128 @us2ui128(i16 zeroext %0) {
; CHECK-LABEL: us2ui128:
; CHECK:       # %bb.0:
; CHECK-NEXT:    or %s1, 0, (0)1
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = zext i16 %0 to i128
  ret i128 %2
}

; Function Attrs: norecurse nounwind readnone
define i128 @c2i128(i8 signext %0) {
; CHECK-LABEL: c2i128:
; CHECK:       # %bb.0:
; CHECK-NEXT:    sra.l %s1, %s0, 63
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = sext i8 %0 to i128
  ret i128 %2
}

; Function Attrs: norecurse nounwind readnone
define i128 @char2ui128(i8 signext %0) {
; CHECK-LABEL: char2ui128:
; CHECK:       # %bb.0:
; CHECK-NEXT:    sra.l %s1, %s0, 63
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = sext i8 %0 to i128
  ret i128 %2
}

; Function Attrs: norecurse nounwind readnone
define i128 @uc2i128(i8 zeroext %0) {
; CHECK-LABEL: uc2i128:
; CHECK:       # %bb.0:
; CHECK-NEXT:    or %s1, 0, (0)1
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = zext i8 %0 to i128
  ret i128 %2
}

; Function Attrs: norecurse nounwind readnone
define i128 @uc2ui128(i8 zeroext %0) {
; CHECK-LABEL: uc2ui128:
; CHECK:       # %bb.0:
; CHECK-NEXT:    or %s1, 0, (0)1
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = zext i8 %0 to i128
  ret i128 %2
}
