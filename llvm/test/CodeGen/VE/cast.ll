; RUN: llc < %s -mtriple=ve-unknown-unknown | FileCheck %s

define i32 @i() {
; CHECK-LABEL: i:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    lea %s0, -2147483648
; CHECK-NEXT:    or %s11, 0, %s9
  ret i32 -2147483648
}

define i32 @ui() {
; CHECK-LABEL: ui:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    lea %s0, -2147483648
; CHECK-NEXT:    or %s11, 0, %s9
  ret i32 -2147483648
}

define i64 @ll() {
; CHECK-LABEL: ll:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    lea %s0, -2147483648
; CHECK-NEXT:    or %s11, 0, %s9
  ret i64 -2147483648
}

define i64 @ull() {
; CHECK-LABEL: ull:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    lea %s0, -2147483648
; CHECK-NEXT:    and %s0, %s0, (32)0
; CHECK-NEXT:    or %s11, 0, %s9
  ret i64 2147483648
}

define double @d2d(double returned %0) {
; CHECK-LABEL: d2d:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    or %s11, 0, %s9
  ret double %0
}

define float @f2f(float returned %0) {
; CHECK-LABEL: f2f:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    or %s11, 0, %s9
  ret float %0
}
define signext i8 @ll2c(i64 %0) {
; CHECK-LABEL: ll2c:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    sll %s0, %s0, 56
; CHECK-NEXT:    sra.l %s0, %s0, 56
; CHECK-NEXT:    # kill: def $sw0 killed $sw0 killed $sx0
; CHECK-NEXT:    or %s11, 0, %s9
  %2 = trunc i64 %0 to i8
  ret i8 %2
}

define zeroext i8 @ll2uc(i64 %0) {
; CHECK-LABEL: ll2uc:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    and %s0, %s0, (56)0
; CHECK-NEXT:    or %s11, 0, %s9
  %2 = trunc i64 %0 to i8
  ret i8 %2
}

define signext i16 @ll2s(i64 %0) {
; CHECK-LABEL: ll2s:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    sll %s0, %s0, 48
; CHECK-NEXT:    sra.l %s0, %s0, 48
; CHECK-NEXT:    # kill: def $sw0 killed $sw0 killed $sx0
; CHECK-NEXT:    or %s11, 0, %s9
  %2 = trunc i64 %0 to i16
  ret i16 %2
}

define zeroext i16 @ll2us(i64 %0) {
; CHECK-LABEL: ll2us:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    and %s0, %s0, (48)0
; CHECK-NEXT:    or %s11, 0, %s9
  %2 = trunc i64 %0 to i16
  ret i16 %2
}

define i32 @ll2i(i64 %0) {
; CHECK-LABEL: ll2i:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    adds.w.sx %s0, %s0, (0)1
; CHECK-NEXT:    or %s11, 0, %s9
  %2 = trunc i64 %0 to i32
  ret i32 %2
}

define i32 @ll2ui(i64 %0) {
; CHECK-LABEL: ll2ui:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    adds.w.sx %s0, %s0, (0)1
; CHECK-NEXT:    or %s11, 0, %s9
  %2 = trunc i64 %0 to i32
  ret i32 %2
}

define i64 @ll2ll(i64 returned %0) {
; CHECK-LABEL: ll2ll:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    or %s11, 0, %s9
  ret i64 %0
}

define i64 @ll2ull(i64 returned %0) {
; CHECK-LABEL: ll2ull:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    or %s11, 0, %s9
  ret i64 %0
}

define signext i8 @ull2c(i64 %0) {
; CHECK-LABEL: ull2c:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    sll %s0, %s0, 56
; CHECK-NEXT:    sra.l %s0, %s0, 56
; CHECK-NEXT:    # kill: def $sw0 killed $sw0 killed $sx0
; CHECK-NEXT:    or %s11, 0, %s9
  %2 = trunc i64 %0 to i8
  ret i8 %2
}

define zeroext i8 @ull2uc(i64 %0) {
; CHECK-LABEL: ull2uc:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    and %s0, %s0, (56)0
; CHECK-NEXT:    or %s11, 0, %s9
  %2 = trunc i64 %0 to i8
  ret i8 %2
}

define signext i16 @ull2s(i64 %0) {
; CHECK-LABEL: ull2s:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    sll %s0, %s0, 48
; CHECK-NEXT:    sra.l %s0, %s0, 48
; CHECK-NEXT:    # kill: def $sw0 killed $sw0 killed $sx0
; CHECK-NEXT:    or %s11, 0, %s9
  %2 = trunc i64 %0 to i16
  ret i16 %2
}

define zeroext i16 @ull2us(i64 %0) {
; CHECK-LABEL: ull2us:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    and %s0, %s0, (48)0
; CHECK-NEXT:    or %s11, 0, %s9
  %2 = trunc i64 %0 to i16
  ret i16 %2
}

define i32 @ull2i(i64 %0) {
; CHECK-LABEL: ull2i:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    adds.w.sx %s0, %s0, (0)1
; CHECK-NEXT:    or %s11, 0, %s9
  %2 = trunc i64 %0 to i32
  ret i32 %2
}

define i32 @ull2ui(i64 %0) {
; CHECK-LABEL: ull2ui:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    adds.w.sx %s0, %s0, (0)1
; CHECK-NEXT:    or %s11, 0, %s9
  %2 = trunc i64 %0 to i32
  ret i32 %2
}

define i64 @ull2ll(i64 returned %0) {
; CHECK-LABEL: ull2ll:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    or %s11, 0, %s9
  ret i64 %0
}

define i64 @ull2ull(i64 returned %0) {
; CHECK-LABEL: ull2ull:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    or %s11, 0, %s9
  ret i64 %0
}

define signext i8 @i2c(i32 %0) {
; CHECK-LABEL: i2c:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    sla.w.sx %s0, %s0, 24
; CHECK-NEXT:    sra.w.sx %s0, %s0, 24
; CHECK-NEXT:    or %s11, 0, %s9
  %2 = trunc i32 %0 to i8
  ret i8 %2
}

define zeroext i8 @i2uc(i32 %0) {
; CHECK-LABEL: i2uc:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    and %s0, %s0, (56)0
; CHECK-NEXT:    or %s11, 0, %s9
  %2 = trunc i32 %0 to i8
  ret i8 %2
}

define signext i16 @i2s(i32 %0) {
; CHECK-LABEL: i2s:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    sla.w.sx %s0, %s0, 16
; CHECK-NEXT:    sra.w.sx %s0, %s0, 16
; CHECK-NEXT:    or %s11, 0, %s9
  %2 = trunc i32 %0 to i16
  ret i16 %2
}

define zeroext i16 @i2us(i32 %0) {
; CHECK-LABEL: i2us:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    and %s0, %s0, (48)0
; CHECK-NEXT:    or %s11, 0, %s9
  %2 = trunc i32 %0 to i16
  ret i16 %2
}

define i32 @i2i(i32 returned %0) {
; CHECK-LABEL: i2i:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    or %s11, 0, %s9
  ret i32 %0
}

define i32 @i2ui(i32 returned %0) {
; CHECK-LABEL: i2ui:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    or %s11, 0, %s9
  ret i32 %0
}

define i64 @i2ll(i32 %0) {
; CHECK-LABEL: i2ll:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    adds.w.sx %s0, %s0, (0)1
; CHECK-NEXT:    or %s11, 0, %s9
  %2 = sext i32 %0 to i64
  ret i64 %2
}

define i64 @i2ull(i32 %0) {
; CHECK-LABEL: i2ull:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    adds.w.sx %s0, %s0, (0)1
; CHECK-NEXT:    or %s11, 0, %s9
  %2 = sext i32 %0 to i64
  ret i64 %2
}

define signext i8 @ui2c(i32 %0) {
; CHECK-LABEL: ui2c:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    sla.w.sx %s0, %s0, 24
; CHECK-NEXT:    sra.w.sx %s0, %s0, 24
; CHECK-NEXT:    or %s11, 0, %s9
  %2 = trunc i32 %0 to i8
  ret i8 %2
}

define zeroext i8 @ui2uc(i32 %0) {
; CHECK-LABEL: ui2uc:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    and %s0, %s0, (56)0
; CHECK-NEXT:    or %s11, 0, %s9
  %2 = trunc i32 %0 to i8
  ret i8 %2
}

define signext i16 @ui2s(i32 %0) {
; CHECK-LABEL: ui2s:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    sla.w.sx %s0, %s0, 16
; CHECK-NEXT:    sra.w.sx %s0, %s0, 16
; CHECK-NEXT:    or %s11, 0, %s9
  %2 = trunc i32 %0 to i16
  ret i16 %2
}

define zeroext i16 @ui2us(i32 %0) {
; CHECK-LABEL: ui2us:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    and %s0, %s0, (48)0
; CHECK-NEXT:    or %s11, 0, %s9
  %2 = trunc i32 %0 to i16
  ret i16 %2
}

define i32 @ui2i(i32 returned %0) {
; CHECK-LABEL: ui2i:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    or %s11, 0, %s9
  ret i32 %0
}

define i32 @ui2ui(i32 returned %0) {
; CHECK-LABEL: ui2ui:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    or %s11, 0, %s9
  ret i32 %0
}

define i64 @ui2ll(i32 %0) {
; CHECK-LABEL: ui2ll:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    adds.w.zx %s0, %s0, (0)1
; CHECK-NEXT:    or %s11, 0, %s9
  %2 = zext i32 %0 to i64
  ret i64 %2
}

define i64 @ui2ull(i32 %0) {
; CHECK-LABEL: ui2ull:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    adds.w.zx %s0, %s0, (0)1
; CHECK-NEXT:    or %s11, 0, %s9
  %2 = zext i32 %0 to i64
  ret i64 %2
}

define signext i8 @s2c(i16 signext %0) {
; CHECK-LABEL: s2c:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    sla.w.sx %s0, %s0, 24
; CHECK-NEXT:    sra.w.sx %s0, %s0, 24
; CHECK-NEXT:    or %s11, 0, %s9
  %2 = trunc i16 %0 to i8
  ret i8 %2
}

define zeroext i8 @s2uc(i16 signext %0) {
; CHECK-LABEL: s2uc:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    and %s0, %s0, (56)0
; CHECK-NEXT:    or %s11, 0, %s9
  %2 = trunc i16 %0 to i8
  ret i8 %2
}

define signext i16 @s2s(i16 returned signext %0) {
; CHECK-LABEL: s2s:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    or %s11, 0, %s9
  ret i16 %0
}

define zeroext i16 @s2us(i16 returned signext %0) {
; CHECK-LABEL: s2us:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    and %s0, %s0, (48)0
; CHECK-NEXT:    or %s11, 0, %s9
  ret i16 %0
}

define i32 @s2i(i16 signext %0) {
; CHECK-LABEL: s2i:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    or %s11, 0, %s9
  %2 = sext i16 %0 to i32
  ret i32 %2
}

define i32 @s2ui(i16 signext %0) {
; CHECK-LABEL: s2ui:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    or %s11, 0, %s9
  %2 = sext i16 %0 to i32
  ret i32 %2
}

define i64 @s2ll(i16 signext %0) {
; CHECK-LABEL: s2ll:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    adds.w.sx %s0, %s0, (0)1
; CHECK-NEXT:    or %s11, 0, %s9
  %2 = sext i16 %0 to i64
  ret i64 %2
}

define i64 @s2ull(i16 signext %0) {
; CHECK-LABEL: s2ull:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    adds.w.sx %s0, %s0, (0)1
; CHECK-NEXT:    or %s11, 0, %s9
  %2 = sext i16 %0 to i64
  ret i64 %2
}

define signext i8 @us2c(i16 zeroext %0) {
; CHECK-LABEL: us2c:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    sla.w.sx %s0, %s0, 24
; CHECK-NEXT:    sra.w.sx %s0, %s0, 24
; CHECK-NEXT:    or %s11, 0, %s9
  %2 = trunc i16 %0 to i8
  ret i8 %2
}

define zeroext i8 @us2uc(i16 zeroext %0) {
; CHECK-LABEL: us2uc:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    and %s0, %s0, (56)0
; CHECK-NEXT:    or %s11, 0, %s9
  %2 = trunc i16 %0 to i8
  ret i8 %2
}

define signext i16 @us2s(i16 returned zeroext %0) {
; CHECK-LABEL: us2s:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    sla.w.sx %s0, %s0, 16
; CHECK-NEXT:    sra.w.sx %s0, %s0, 16
; CHECK-NEXT:    or %s11, 0, %s9
  ret i16 %0
}

define zeroext i16 @us2us(i16 returned zeroext %0) {
; CHECK-LABEL: us2us:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    or %s11, 0, %s9
  ret i16 %0
}

define i32 @us2i(i16 zeroext %0) {
; CHECK-LABEL: us2i:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    or %s11, 0, %s9
  %2 = zext i16 %0 to i32
  ret i32 %2
}

define i32 @us2ui(i16 zeroext %0) {
; CHECK-LABEL: us2ui:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    or %s11, 0, %s9
  %2 = zext i16 %0 to i32
  ret i32 %2
}

define i64 @us2ll(i16 zeroext %0) {
; CHECK-LABEL: us2ll:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    adds.w.zx %s0, %s0, (0)1
; CHECK-NEXT:    or %s11, 0, %s9
  %2 = zext i16 %0 to i64
  ret i64 %2
}

define i64 @us2ull(i16 zeroext %0) {
; CHECK-LABEL: us2ull:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    adds.w.zx %s0, %s0, (0)1
; CHECK-NEXT:    or %s11, 0, %s9
  %2 = zext i16 %0 to i64
  ret i64 %2
}

define signext i8 @c2c(i8 returned signext %0) {
; CHECK-LABEL: c2c:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    or %s11, 0, %s9
  ret i8 %0
}

define zeroext i8 @c2uc(i8 returned signext %0) {
; CHECK-LABEL: c2uc:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    and %s0, %s0, (56)0
; CHECK-NEXT:    or %s11, 0, %s9
  ret i8 %0
}

define signext i16 @c2s(i8 signext %0) {
; CHECK-LABEL: c2s:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    or %s11, 0, %s9
  %2 = sext i8 %0 to i16
  ret i16 %2
}

define zeroext i16 @c2us(i8 signext %0) {
; CHECK-LABEL: c2us:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    and %s0, %s0, (48)0
; CHECK-NEXT:    or %s11, 0, %s9
  %2 = sext i8 %0 to i16
  ret i16 %2
}

define i32 @c2i(i8 signext %0) {
; CHECK-LABEL: c2i:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    or %s11, 0, %s9
  %2 = sext i8 %0 to i32
  ret i32 %2
}

define i32 @c2ui(i8 signext %0) {
; CHECK-LABEL: c2ui:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    or %s11, 0, %s9
  %2 = sext i8 %0 to i32
  ret i32 %2
}

define i64 @c2ll(i8 signext %0) {
; CHECK-LABEL: c2ll:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    adds.w.sx %s0, %s0, (0)1
; CHECK-NEXT:    or %s11, 0, %s9
  %2 = sext i8 %0 to i64
  ret i64 %2
}

define i64 @c2ull(i8 signext %0) {
; CHECK-LABEL: c2ull:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    adds.w.sx %s0, %s0, (0)1
; CHECK-NEXT:    or %s11, 0, %s9
  %2 = sext i8 %0 to i64
  ret i64 %2
}

define signext i8 @uc2c(i8 returned zeroext %0) {
; CHECK-LABEL: uc2c:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    sla.w.sx %s0, %s0, 24
; CHECK-NEXT:    sra.w.sx %s0, %s0, 24
; CHECK-NEXT:    or %s11, 0, %s9
  ret i8 %0
}

define zeroext i8 @uc2uc(i8 returned zeroext %0) {
; CHECK-LABEL: uc2uc:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    or %s11, 0, %s9
  ret i8 %0
}

define signext i16 @uc2s(i8 zeroext %0) {
; CHECK-LABEL: uc2s:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    or %s11, 0, %s9
  %2 = zext i8 %0 to i16
  ret i16 %2
}

define zeroext i16 @uc2us(i8 zeroext %0) {
; CHECK-LABEL: uc2us:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    or %s11, 0, %s9
  %2 = zext i8 %0 to i16
  ret i16 %2
}

define i32 @uc2i(i8 zeroext %0) {
; CHECK-LABEL: uc2i:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    or %s11, 0, %s9
  %2 = zext i8 %0 to i32
  ret i32 %2
}

define i32 @uc2ui(i8 zeroext %0) {
; CHECK-LABEL: uc2ui:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    or %s11, 0, %s9
  %2 = zext i8 %0 to i32
  ret i32 %2
}

define i64 @uc2ll(i8 zeroext %0) {
; CHECK-LABEL: uc2ll:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    adds.w.zx %s0, %s0, (0)1
; CHECK-NEXT:    or %s11, 0, %s9
  %2 = zext i8 %0 to i64
  ret i64 %2
}

define i64 @uc2ull(i8 zeroext %0) {
; CHECK-LABEL: uc2ull:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    adds.w.zx %s0, %s0, (0)1
; CHECK-NEXT:    or %s11, 0, %s9
  %2 = zext i8 %0 to i64
  ret i64 %2
}

; Function Attrs: norecurse nounwind readnone
define i128 @i128() {
; CHECK-LABEL: i128:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    lea %s0, -2147483648
; CHECK-NEXT:    or %s1, -1, (0)1
; CHECK-NEXT:    or %s11, 0, %s9
  ret i128 -2147483648
}

; Function Attrs: norecurse nounwind readnone
define i128 @ui128() {
; CHECK-LABEL: ui128:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    lea %s0, -2147483648
; CHECK-NEXT:    or %s1, -1, (0)1
; CHECK-NEXT:    or %s11, 0, %s9
  ret i128 -2147483648
}

; Function Attrs: norecurse nounwind readnone
define signext i8 @i1282c(i128 %0) {
; CHECK-LABEL: i1282c:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    sll %s0, %s0, 56
; CHECK-NEXT:    sra.l %s0, %s0, 56
; CHECK-NEXT:    # kill: def $sw0 killed $sw0 killed $sx0
; CHECK-NEXT:    or %s11, 0, %s9
  %2 = trunc i128 %0 to i8
  ret i8 %2
}

; Function Attrs: norecurse nounwind readnone
define signext i8 @ui1282c(i128 %0) {
; CHECK-LABEL: ui1282c:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    sll %s0, %s0, 56
; CHECK-NEXT:    sra.l %s0, %s0, 56
; CHECK-NEXT:    # kill: def $sw0 killed $sw0 killed $sx0
; CHECK-NEXT:    or %s11, 0, %s9
  %2 = trunc i128 %0 to i8
  ret i8 %2
}

; Function Attrs: norecurse nounwind readnone
define zeroext i8 @i1282uc(i128 %0) {
; CHECK-LABEL: i1282uc:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    and %s0, %s0, (56)0
; CHECK-NEXT:    or %s11, 0, %s9
  %2 = trunc i128 %0 to i8
  ret i8 %2
}

; Function Attrs: norecurse nounwind readnone
define zeroext i8 @ui1282uc(i128 %0) {
; CHECK-LABEL: ui1282uc:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    and %s0, %s0, (56)0
; CHECK-NEXT:    or %s11, 0, %s9
  %2 = trunc i128 %0 to i8
  ret i8 %2
}

; Function Attrs: norecurse nounwind readnone
define signext i16 @i1282s(i128 %0) {
; CHECK-LABEL: i1282s:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    sll %s0, %s0, 48
; CHECK-NEXT:    sra.l %s0, %s0, 48
; CHECK-NEXT:    # kill: def $sw0 killed $sw0 killed $sx0
; CHECK-NEXT:    or %s11, 0, %s9
  %2 = trunc i128 %0 to i16
  ret i16 %2
}

; Function Attrs: norecurse nounwind readnone
define signext i16 @ui1282s(i128 %0) {
; CHECK-LABEL: ui1282s:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    sll %s0, %s0, 48
; CHECK-NEXT:    sra.l %s0, %s0, 48
; CHECK-NEXT:    # kill: def $sw0 killed $sw0 killed $sx0
; CHECK-NEXT:    or %s11, 0, %s9
  %2 = trunc i128 %0 to i16
  ret i16 %2
}

; Function Attrs: norecurse nounwind readnone
define zeroext i16 @i1282us(i128 %0) {
; CHECK-LABEL: i1282us:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    and %s0, %s0, (48)0
; CHECK-NEXT:    or %s11, 0, %s9
  %2 = trunc i128 %0 to i16
  ret i16 %2
}

; Function Attrs: norecurse nounwind readnone
define zeroext i16 @ui1282us(i128 %0) {
; CHECK-LABEL: ui1282us:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    and %s0, %s0, (48)0
; CHECK-NEXT:    or %s11, 0, %s9
  %2 = trunc i128 %0 to i16
  ret i16 %2
}

; Function Attrs: norecurse nounwind readnone
define i32 @i1282i(i128 %0) {
; CHECK-LABEL: i1282i:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    adds.w.sx %s0, %s0, (0)1
; CHECK-NEXT:    or %s11, 0, %s9
  %2 = trunc i128 %0 to i32
  ret i32 %2
}

; Function Attrs: norecurse nounwind readnone
define i32 @ui1282i(i128 %0) {
; CHECK-LABEL: ui1282i:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    adds.w.sx %s0, %s0, (0)1
; CHECK-NEXT:    or %s11, 0, %s9
  %2 = trunc i128 %0 to i32
  ret i32 %2
}

; Function Attrs: norecurse nounwind readnone
define i32 @i1282ui(i128 %0) {
; CHECK-LABEL: i1282ui:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    adds.w.sx %s0, %s0, (0)1
; CHECK-NEXT:    or %s11, 0, %s9
  %2 = trunc i128 %0 to i32
  ret i32 %2
}

; Function Attrs: norecurse nounwind readnone
define i32 @ui1282ui(i128 %0) {
; CHECK-LABEL: ui1282ui:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    adds.w.sx %s0, %s0, (0)1
; CHECK-NEXT:    or %s11, 0, %s9
  %2 = trunc i128 %0 to i32
  ret i32 %2
}

; Function Attrs: norecurse nounwind readnone
define i64 @i1282ll(i128 %0) {
; CHECK-LABEL: i1282ll:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    or %s11, 0, %s9
  %2 = trunc i128 %0 to i64
  ret i64 %2
}

; Function Attrs: norecurse nounwind readnone
define i64 @ui1282ll(i128 %0) {
; CHECK-LABEL: ui1282ll:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    or %s11, 0, %s9
  %2 = trunc i128 %0 to i64
  ret i64 %2
}

; Function Attrs: norecurse nounwind readnone
define i64 @i1282ull(i128 %0) {
; CHECK-LABEL: i1282ull:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    or %s11, 0, %s9
  %2 = trunc i128 %0 to i64
  ret i64 %2
}

; Function Attrs: norecurse nounwind readnone
define i64 @ui1282ull(i128 %0) {
; CHECK-LABEL: ui1282ull:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    or %s11, 0, %s9
  %2 = trunc i128 %0 to i64
  ret i64 %2
}

; Function Attrs: norecurse nounwind readnone
define i128 @i1282ui128(i128 returned %0) {
; CHECK-LABEL: i1282ui128:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    or %s11, 0, %s9
  ret i128 %0
}

; Function Attrs: norecurse nounwind readnone
define i128 @ui1282i128(i128 returned %0) {
; CHECK-LABEL: ui1282i128:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    or %s11, 0, %s9
  ret i128 %0
}

; Function Attrs: norecurse nounwind readnone
define i128 @ll2i128(i64 %0) {
; CHECK-LABEL: ll2i128:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    sra.l %s1, %s0, 63
; CHECK-NEXT:    or %s11, 0, %s9
  %2 = sext i64 %0 to i128
  ret i128 %2
}

; Function Attrs: norecurse nounwind readnone
define i128 @ll2ui128(i64 %0) {
; CHECK-LABEL: ll2ui128:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    sra.l %s1, %s0, 63
; CHECK-NEXT:    or %s11, 0, %s9
  %2 = sext i64 %0 to i128
  ret i128 %2
}

; Function Attrs: norecurse nounwind readnone
define i128 @ull2i128(i64 %0) {
; CHECK-LABEL: ull2i128:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    or %s1, 0, (0)1
; CHECK-NEXT:    or %s11, 0, %s9
  %2 = zext i64 %0 to i128
  ret i128 %2
}

; Function Attrs: norecurse nounwind readnone
define i128 @ull2ui128(i64 %0) {
; CHECK-LABEL: ull2ui128:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    or %s1, 0, (0)1
; CHECK-NEXT:    or %s11, 0, %s9
  %2 = zext i64 %0 to i128
  ret i128 %2
}

; Function Attrs: norecurse nounwind readnone
define i128 @i2i128(i32 %0) {
; CHECK-LABEL: i2i128:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    adds.w.sx %s0, %s0, (0)1
; CHECK-NEXT:    sra.l %s1, %s0, 63
; CHECK-NEXT:    or %s11, 0, %s9
  %2 = sext i32 %0 to i128
  ret i128 %2
}

; Function Attrs: norecurse nounwind readnone
define i128 @i2ui128(i32 %0) {
; CHECK-LABEL: i2ui128:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    adds.w.sx %s0, %s0, (0)1
; CHECK-NEXT:    sra.l %s1, %s0, 63
; CHECK-NEXT:    or %s11, 0, %s9
  %2 = sext i32 %0 to i128
  ret i128 %2
}

; Function Attrs: norecurse nounwind readnone
define i128 @ui2i128(i32 %0) {
; CHECK-LABEL: ui2i128:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    adds.w.zx %s0, %s0, (0)1
; CHECK-NEXT:    or %s1, 0, (0)1
; CHECK-NEXT:    or %s11, 0, %s9
  %2 = zext i32 %0 to i128
  ret i128 %2
}

; Function Attrs: norecurse nounwind readnone
define i128 @ui2ui128(i32 %0) {
; CHECK-LABEL: ui2ui128:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    adds.w.zx %s0, %s0, (0)1
; CHECK-NEXT:    or %s1, 0, (0)1
; CHECK-NEXT:    or %s11, 0, %s9
  %2 = zext i32 %0 to i128
  ret i128 %2
}

; Function Attrs: norecurse nounwind readnone
define i128 @s2i128(i16 signext %0) {
; CHECK-LABEL: s2i128:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    adds.w.sx %s0, %s0, (0)1
; CHECK-NEXT:    sra.l %s1, %s0, 63
; CHECK-NEXT:    or %s11, 0, %s9
  %2 = sext i16 %0 to i128
  ret i128 %2
}

; Function Attrs: norecurse nounwind readnone
define i128 @s2ui128(i16 signext %0) {
; CHECK-LABEL: s2ui128:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    adds.w.sx %s0, %s0, (0)1
; CHECK-NEXT:    sra.l %s1, %s0, 63
; CHECK-NEXT:    or %s11, 0, %s9
  %2 = sext i16 %0 to i128
  ret i128 %2
}

; Function Attrs: norecurse nounwind readnone
define i128 @us2i128(i16 zeroext %0) {
; CHECK-LABEL: us2i128:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    adds.w.zx %s0, %s0, (0)1
; CHECK-NEXT:    or %s1, 0, (0)1
; CHECK-NEXT:    or %s11, 0, %s9
  %2 = zext i16 %0 to i128
  ret i128 %2
}

; Function Attrs: norecurse nounwind readnone
define i128 @us2ui128(i16 zeroext %0) {
; CHECK-LABEL: us2ui128:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    adds.w.zx %s0, %s0, (0)1
; CHECK-NEXT:    or %s1, 0, (0)1
; CHECK-NEXT:    or %s11, 0, %s9
  %2 = zext i16 %0 to i128
  ret i128 %2
}

; Function Attrs: norecurse nounwind readnone
define i128 @c2i128(i8 signext %0) {
; CHECK-LABEL: c2i128:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    adds.w.sx %s0, %s0, (0)1
; CHECK-NEXT:    sra.l %s1, %s0, 63
; CHECK-NEXT:    or %s11, 0, %s9
  %2 = sext i8 %0 to i128
  ret i128 %2
}

; Function Attrs: norecurse nounwind readnone
define i128 @char2ui128(i8 signext %0) {
; CHECK-LABEL: char2ui128:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    adds.w.sx %s0, %s0, (0)1
; CHECK-NEXT:    sra.l %s1, %s0, 63
; CHECK-NEXT:    or %s11, 0, %s9
  %2 = sext i8 %0 to i128
  ret i128 %2
}

; Function Attrs: norecurse nounwind readnone
define i128 @uc2i128(i8 zeroext %0) {
; CHECK-LABEL: uc2i128:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    adds.w.zx %s0, %s0, (0)1
; CHECK-NEXT:    or %s1, 0, (0)1
; CHECK-NEXT:    or %s11, 0, %s9
  %2 = zext i8 %0 to i128
  ret i128 %2
}

; Function Attrs: norecurse nounwind readnone
define i128 @uc2ui128(i8 zeroext %0) {
; CHECK-LABEL: uc2ui128:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    adds.w.zx %s0, %s0, (0)1
; CHECK-NEXT:    or %s1, 0, (0)1
; CHECK-NEXT:    or %s11, 0, %s9
  %2 = zext i8 %0 to i128
  ret i128 %2
}
