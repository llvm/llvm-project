; Test insertions of memory into the low byte of an i64.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

; Check a plain insertion with (or (and ... -0xff) (zext (load ....))).
; The whole sequence can be performed by IC.
define i64 @f1(i64 %orig, ptr %ptr) {
; CHECK-LABEL: f1:
; CHECK-NOT: ni
; CHECK: ic %r2, 0(%r3)
; CHECK: br %r14
  %val = load i8, ptr %ptr
  %ptr2 = zext i8 %val to i64
  %ptr1 = and i64 %orig, -256
  %or = or i64 %ptr1, %ptr2
  ret i64 %or
}

; Like f1, but with the operands reversed.
define i64 @f2(i64 %orig, ptr %ptr) {
; CHECK-LABEL: f2:
; CHECK-NOT: ni
; CHECK: ic %r2, 0(%r3)
; CHECK: br %r14
  %val = load i8, ptr %ptr
  %ptr2 = zext i8 %val to i64
  %ptr1 = and i64 %orig, -256
  %or = or i64 %ptr2, %ptr1
  ret i64 %or
}

; Check a case where more bits than lower 8 are masked out of the
; register value.  We can use IC but must keep the original mask.
define i64 @f3(i64 %orig, ptr %ptr) {
; CHECK-LABEL: f3:
; CHECK: nill %r2, 65024
; CHECK: ic %r2, 0(%r3)
; CHECK: br %r14
  %val = load i8, ptr %ptr
  %ptr2 = zext i8 %val to i64
  %ptr1 = and i64 %orig, -512
  %or = or i64 %ptr1, %ptr2
  ret i64 %or
}

; Like f3, but with the operands reversed.
define i64 @f4(i64 %orig, ptr %ptr) {
; CHECK-LABEL: f4:
; CHECK: nill %r2, 65024
; CHECK: ic %r2, 0(%r3)
; CHECK: br %r14
  %val = load i8, ptr %ptr
  %ptr2 = zext i8 %val to i64
  %ptr1 = and i64 %orig, -512
  %or = or i64 %ptr2, %ptr1
  ret i64 %or
}

; Check a case where the low 8 bits are cleared by a shift left.
define i64 @f5(i64 %orig, ptr %ptr) {
; CHECK-LABEL: f5:
; CHECK: sllg %r2, %r2, 8
; CHECK: ic %r2, 0(%r3)
; CHECK: br %r14
  %val = load i8, ptr %ptr
  %ptr2 = zext i8 %val to i64
  %ptr1 = shl i64 %orig, 8
  %or = or i64 %ptr1, %ptr2
  ret i64 %or
}

; Like f5, but with the operands reversed.
define i64 @f6(i64 %orig, ptr %ptr) {
; CHECK-LABEL: f6:
; CHECK: sllg %r2, %r2, 8
; CHECK: ic %r2, 0(%r3)
; CHECK: br %r14
  %val = load i8, ptr %ptr
  %ptr2 = zext i8 %val to i64
  %ptr1 = shl i64 %orig, 8
  %or = or i64 %ptr2, %ptr1
  ret i64 %or
}

; Check insertions into a constant.
define i64 @f7(i64 %orig, ptr %ptr) {
; CHECK-LABEL: f7:
; CHECK: lghi %r2, 256
; CHECK: ic %r2, 0(%r3)
; CHECK: br %r14
  %val = load i8, ptr %ptr
  %ptr2 = zext i8 %val to i64
  %or = or i64 %ptr2, 256
  ret i64 %or
}

; Like f7, but with the operands reversed.
define i64 @f8(i64 %orig, ptr %ptr) {
; CHECK-LABEL: f8:
; CHECK: lghi %r2, 256
; CHECK: ic %r2, 0(%r3)
; CHECK: br %r14
  %val = load i8, ptr %ptr
  %ptr2 = zext i8 %val to i64
  %or = or i64 256, %ptr2
  ret i64 %or
}

; Check the high end of the IC range.
define i64 @f9(i64 %orig, ptr %src) {
; CHECK-LABEL: f9:
; CHECK: ic %r2, 4095(%r3)
; CHECK: br %r14
  %ptr = getelementptr i8, ptr %src, i64 4095
  %val = load i8, ptr %ptr
  %src2 = zext i8 %val to i64
  %src1 = and i64 %orig, -256
  %or = or i64 %src2, %src1
  ret i64 %or
}

; Check the next byte up, which should use ICY instead of IC.
define i64 @f10(i64 %orig, ptr %src) {
; CHECK-LABEL: f10:
; CHECK: icy %r2, 4096(%r3)
; CHECK: br %r14
  %ptr = getelementptr i8, ptr %src, i64 4096
  %val = load i8, ptr %ptr
  %src2 = zext i8 %val to i64
  %src1 = and i64 %orig, -256
  %or = or i64 %src2, %src1
  ret i64 %or
}

; Check the high end of the ICY range.
define i64 @f11(i64 %orig, ptr %src) {
; CHECK-LABEL: f11:
; CHECK: icy %r2, 524287(%r3)
; CHECK: br %r14
  %ptr = getelementptr i8, ptr %src, i64 524287
  %val = load i8, ptr %ptr
  %src2 = zext i8 %val to i64
  %src1 = and i64 %orig, -256
  %or = or i64 %src2, %src1
  ret i64 %or
}

; Check the next byte up, which needs separate address logic.
; Other sequences besides this one would be OK.
define i64 @f12(i64 %orig, ptr %src) {
; CHECK-LABEL: f12:
; CHECK: agfi %r3, 524288
; CHECK: ic %r2, 0(%r3)
; CHECK: br %r14
  %ptr = getelementptr i8, ptr %src, i64 524288
  %val = load i8, ptr %ptr
  %src2 = zext i8 %val to i64
  %src1 = and i64 %orig, -256
  %or = or i64 %src2, %src1
  ret i64 %or
}

; Check the high end of the negative ICY range.
define i64 @f13(i64 %orig, ptr %src) {
; CHECK-LABEL: f13:
; CHECK: icy %r2, -1(%r3)
; CHECK: br %r14
  %ptr = getelementptr i8, ptr %src, i64 -1
  %val = load i8, ptr %ptr
  %src2 = zext i8 %val to i64
  %src1 = and i64 %orig, -256
  %or = or i64 %src2, %src1
  ret i64 %or
}

; Check the low end of the ICY range.
define i64 @f14(i64 %orig, ptr %src) {
; CHECK-LABEL: f14:
; CHECK: icy %r2, -524288(%r3)
; CHECK: br %r14
  %ptr = getelementptr i8, ptr %src, i64 -524288
  %val = load i8, ptr %ptr
  %src2 = zext i8 %val to i64
  %src1 = and i64 %orig, -256
  %or = or i64 %src2, %src1
  ret i64 %or
}

; Check the next byte down, which needs separate address logic.
; Other sequences besides this one would be OK.
define i64 @f15(i64 %orig, ptr %src) {
; CHECK-LABEL: f15:
; CHECK: agfi %r3, -524289
; CHECK: ic %r2, 0(%r3)
; CHECK: br %r14
  %ptr = getelementptr i8, ptr %src, i64 -524289
  %val = load i8, ptr %ptr
  %src2 = zext i8 %val to i64
  %src1 = and i64 %orig, -256
  %or = or i64 %src2, %src1
  ret i64 %or
}

; Check that IC allows an index.
define i64 @f16(i64 %orig, ptr %src, i64 %index) {
; CHECK-LABEL: f16:
; CHECK: ic %r2, 4095({{%r4,%r3|%r3,%r4}})
; CHECK: br %r14
  %ptr1 = getelementptr i8, ptr %src, i64 %index
  %ptr2 = getelementptr i8, ptr %ptr1, i64 4095
  %val = load i8, ptr %ptr2
  %src2 = zext i8 %val to i64
  %src1 = and i64 %orig, -256
  %or = or i64 %src2, %src1
  ret i64 %or
}

; Check that ICY allows an index.
define i64 @f17(i64 %orig, ptr %src, i64 %index) {
; CHECK-LABEL: f17:
; CHECK: icy %r2, 4096({{%r4,%r3|%r3,%r4}})
; CHECK: br %r14
  %ptr1 = getelementptr i8, ptr %src, i64 %index
  %ptr2 = getelementptr i8, ptr %ptr1, i64 4096
  %val = load i8, ptr %ptr2
  %src2 = zext i8 %val to i64
  %src1 = and i64 %orig, -256
  %or = or i64 %src2, %src1
  ret i64 %or
}
