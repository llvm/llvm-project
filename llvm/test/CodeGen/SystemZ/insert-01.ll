; Test insertions of memory into the low byte of an i32.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

; Check a plain insertion with (or (and ... -0xff) (zext (load ....))).
; The whole sequence can be performed by IC.
define i32 @f1(i32 %orig, ptr %ptr) {
; CHECK-LABEL: f1:
; CHECK-NOT: ni
; CHECK: ic %r2, 0(%r3)
; CHECK: br %r14
  %val = load i8, ptr %ptr
  %ptr2 = zext i8 %val to i32
  %ptr1 = and i32 %orig, -256
  %or = or i32 %ptr1, %ptr2
  ret i32 %or
}

; Like f1, but with the operands reversed.
define i32 @f2(i32 %orig, ptr %ptr) {
; CHECK-LABEL: f2:
; CHECK-NOT: ni
; CHECK: ic %r2, 0(%r3)
; CHECK: br %r14
  %val = load i8, ptr %ptr
  %ptr2 = zext i8 %val to i32
  %ptr1 = and i32 %orig, -256
  %or = or i32 %ptr2, %ptr1
  ret i32 %or
}

; Check a case where more bits than lower 8 are masked out of the
; register value.  We can use IC but must keep the original mask.
define i32 @f3(i32 %orig, ptr %ptr) {
; CHECK-LABEL: f3:
; CHECK: nill %r2, 65024
; CHECK: ic %r2, 0(%r3)
; CHECK: br %r14
  %val = load i8, ptr %ptr
  %ptr2 = zext i8 %val to i32
  %ptr1 = and i32 %orig, -512
  %or = or i32 %ptr1, %ptr2
  ret i32 %or
}

; Like f3, but with the operands reversed.
define i32 @f4(i32 %orig, ptr %ptr) {
; CHECK-LABEL: f4:
; CHECK: nill %r2, 65024
; CHECK: ic %r2, 0(%r3)
; CHECK: br %r14
  %val = load i8, ptr %ptr
  %ptr2 = zext i8 %val to i32
  %ptr1 = and i32 %orig, -512
  %or = or i32 %ptr2, %ptr1
  ret i32 %or
}

; Check a case where the low 8 bits are cleared by a shift left.
define i32 @f5(i32 %orig, ptr %ptr) {
; CHECK-LABEL: f5:
; CHECK: sll %r2, 8
; CHECK: ic %r2, 0(%r3)
; CHECK: br %r14
  %val = load i8, ptr %ptr
  %ptr2 = zext i8 %val to i32
  %ptr1 = shl i32 %orig, 8
  %or = or i32 %ptr1, %ptr2
  ret i32 %or
}

; Like f5, but with the operands reversed.
define i32 @f6(i32 %orig, ptr %ptr) {
; CHECK-LABEL: f6:
; CHECK: sll %r2, 8
; CHECK: ic %r2, 0(%r3)
; CHECK: br %r14
  %val = load i8, ptr %ptr
  %ptr2 = zext i8 %val to i32
  %ptr1 = shl i32 %orig, 8
  %or = or i32 %ptr2, %ptr1
  ret i32 %or
}

; Check insertions into a constant.
define i32 @f7(i32 %orig, ptr %ptr) {
; CHECK-LABEL: f7:
; CHECK: lhi %r2, 256
; CHECK: ic %r2, 0(%r3)
; CHECK: br %r14
  %val = load i8, ptr %ptr
  %ptr2 = zext i8 %val to i32
  %or = or i32 %ptr2, 256
  ret i32 %or
}

; Like f7, but with the operands reversed.
define i32 @f8(i32 %orig, ptr %ptr) {
; CHECK-LABEL: f8:
; CHECK: lhi %r2, 256
; CHECK: ic %r2, 0(%r3)
; CHECK: br %r14
  %val = load i8, ptr %ptr
  %ptr2 = zext i8 %val to i32
  %or = or i32 256, %ptr2
  ret i32 %or
}

; Check the high end of the IC range.
define i32 @f9(i32 %orig, ptr %src) {
; CHECK-LABEL: f9:
; CHECK: ic %r2, 4095(%r3)
; CHECK: br %r14
  %ptr = getelementptr i8, ptr %src, i64 4095
  %val = load i8, ptr %ptr
  %src2 = zext i8 %val to i32
  %src1 = and i32 %orig, -256
  %or = or i32 %src2, %src1
  ret i32 %or
}

; Check the next byte up, which should use ICY instead of IC.
define i32 @f10(i32 %orig, ptr %src) {
; CHECK-LABEL: f10:
; CHECK: icy %r2, 4096(%r3)
; CHECK: br %r14
  %ptr = getelementptr i8, ptr %src, i64 4096
  %val = load i8, ptr %ptr
  %src2 = zext i8 %val to i32
  %src1 = and i32 %orig, -256
  %or = or i32 %src2, %src1
  ret i32 %or
}

; Check the high end of the ICY range.
define i32 @f11(i32 %orig, ptr %src) {
; CHECK-LABEL: f11:
; CHECK: icy %r2, 524287(%r3)
; CHECK: br %r14
  %ptr = getelementptr i8, ptr %src, i64 524287
  %val = load i8, ptr %ptr
  %src2 = zext i8 %val to i32
  %src1 = and i32 %orig, -256
  %or = or i32 %src2, %src1
  ret i32 %or
}

; Check the next byte up, which needs separate address logic.
; Other sequences besides this one would be OK.
define i32 @f12(i32 %orig, ptr %src) {
; CHECK-LABEL: f12:
; CHECK: agfi %r3, 524288
; CHECK: ic %r2, 0(%r3)
; CHECK: br %r14
  %ptr = getelementptr i8, ptr %src, i64 524288
  %val = load i8, ptr %ptr
  %src2 = zext i8 %val to i32
  %src1 = and i32 %orig, -256
  %or = or i32 %src2, %src1
  ret i32 %or
}

; Check the high end of the negative ICY range.
define i32 @f13(i32 %orig, ptr %src) {
; CHECK-LABEL: f13:
; CHECK: icy %r2, -1(%r3)
; CHECK: br %r14
  %ptr = getelementptr i8, ptr %src, i64 -1
  %val = load i8, ptr %ptr
  %src2 = zext i8 %val to i32
  %src1 = and i32 %orig, -256
  %or = or i32 %src2, %src1
  ret i32 %or
}

; Check the low end of the ICY range.
define i32 @f14(i32 %orig, ptr %src) {
; CHECK-LABEL: f14:
; CHECK: icy %r2, -524288(%r3)
; CHECK: br %r14
  %ptr = getelementptr i8, ptr %src, i64 -524288
  %val = load i8, ptr %ptr
  %src2 = zext i8 %val to i32
  %src1 = and i32 %orig, -256
  %or = or i32 %src2, %src1
  ret i32 %or
}

; Check the next byte down, which needs separate address logic.
; Other sequences besides this one would be OK.
define i32 @f15(i32 %orig, ptr %src) {
; CHECK-LABEL: f15:
; CHECK: agfi %r3, -524289
; CHECK: ic %r2, 0(%r3)
; CHECK: br %r14
  %ptr = getelementptr i8, ptr %src, i64 -524289
  %val = load i8, ptr %ptr
  %src2 = zext i8 %val to i32
  %src1 = and i32 %orig, -256
  %or = or i32 %src2, %src1
  ret i32 %or
}

; Check that IC allows an index.
define i32 @f16(i32 %orig, ptr %src, i64 %index) {
; CHECK-LABEL: f16:
; CHECK: ic %r2, 4095({{%r4,%r3|%r3,%r4}})
; CHECK: br %r14
  %ptr1 = getelementptr i8, ptr %src, i64 %index
  %ptr2 = getelementptr i8, ptr %ptr1, i64 4095
  %val = load i8, ptr %ptr2
  %src2 = zext i8 %val to i32
  %src1 = and i32 %orig, -256
  %or = or i32 %src2, %src1
  ret i32 %or
}

; Check that ICY allows an index.
define i32 @f17(i32 %orig, ptr %src, i64 %index) {
; CHECK-LABEL: f17:
; CHECK: icy %r2, 4096({{%r4,%r3|%r3,%r4}})
; CHECK: br %r14
  %ptr1 = getelementptr i8, ptr %src, i64 %index
  %ptr2 = getelementptr i8, ptr %ptr1, i64 4096
  %val = load i8, ptr %ptr2
  %src2 = zext i8 %val to i32
  %src1 = and i32 %orig, -256
  %or = or i32 %src2, %src1
  ret i32 %or
}
