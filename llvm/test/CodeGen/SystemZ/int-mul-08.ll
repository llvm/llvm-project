; Test high-part i64->i128 multiplications.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -asm-verbose=0 | FileCheck %s

declare i64 @foo()

; Check zero-extended multiplication in which only the high part is used.
define i64 @f1(i64 %dummy, i64 %a, i64 %b) {
; CHECK-LABEL: f1:
; CHECK-NOT: {{%r[234]}}
; CHECK: mlgr %r2, %r4
; CHECK: br %r14
  %ax = zext i64 %a to i128
  %bx = zext i64 %b to i128
  %mulx = mul i128 %ax, %bx
  %highx = lshr i128 %mulx, 64
  %high = trunc i128 %highx to i64
  ret i64 %high
}

; Check sign-extended multiplication in which only the high part is used.
; This needs a rather convoluted sequence.
define i64 @f2(i64 %dummy, i64 %a, i64 %b) {
; CHECK-LABEL: f2:
; CHECK-DAG: srag [[RES1:%r[0-5]]], %r3, 63
; CHECK-DAG: srag [[RES2:%r[0-5]]], %r4, 63
; CHECK-DAG: ngr [[RES1]], %r4
; CHECK-DAG: ngr [[RES2]], %r3
; CHECK-DAG: agr [[RES2]], [[RES1]]
; CHECK-DAG: mlgr %r2, %r4
; CHECK: sgr %r2, [[RES2]]
; CHECK: br %r14
  %ax = sext i64 %a to i128
  %bx = sext i64 %b to i128
  %mulx = mul i128 %ax, %bx
  %highx = lshr i128 %mulx, 64
  %high = trunc i128 %highx to i64
  ret i64 %high
}

; Check zero-extended multiplication in which only part of the high half
; is used.
define i64 @f3(i64 %dummy, i64 %a, i64 %b) {
; CHECK-LABEL: f3:
; CHECK-NOT: {{%r[234]}}
; CHECK: mlgr %r2, %r4
; CHECK: srlg %r2, %r2, 3
; CHECK: br %r14
  %ax = zext i64 %a to i128
  %bx = zext i64 %b to i128
  %mulx = mul i128 %ax, %bx
  %highx = lshr i128 %mulx, 67
  %high = trunc i128 %highx to i64
  ret i64 %high
}

; Check zero-extended multiplication in which the result is split into
; high and low halves.
define i64 @f4(i64 %dummy, i64 %a, i64 %b) {
; CHECK-LABEL: f4:
; CHECK-NOT: {{%r[234]}}
; CHECK: mlgr %r2, %r4
; CHECK: ogr %r2, %r3
; CHECK: br %r14
  %ax = zext i64 %a to i128
  %bx = zext i64 %b to i128
  %mulx = mul i128 %ax, %bx
  %highx = lshr i128 %mulx, 64
  %high = trunc i128 %highx to i64
  %low = trunc i128 %mulx to i64
  %or = or i64 %high, %low
  ret i64 %or
}

; Check division by a constant, which should use multiplication instead.
define i64 @f5(i64 %dummy, i64 %a) {
; CHECK-LABEL: f5:
; CHECK: mlgr %r2,
; CHECK: srlg %r2, %r2,
; CHECK: br %r14
  %res = udiv i64 %a, 1234
  ret i64 %res
}

; Check MLG with no displacement.
define i64 @f6(i64 %dummy, i64 %a, i64 *%src) {
; CHECK-LABEL: f6:
; CHECK-NOT: {{%r[234]}}
; CHECK: mlg %r2, 0(%r4)
; CHECK: br %r14
  %b = load i64, i64 *%src
  %ax = zext i64 %a to i128
  %bx = zext i64 %b to i128
  %mulx = mul i128 %ax, %bx
  %highx = lshr i128 %mulx, 64
  %high = trunc i128 %highx to i64
  ret i64 %high
}

; Check the high end of the aligned MLG range.
define i64 @f7(i64 %dummy, i64 %a, i64 *%src) {
; CHECK-LABEL: f7:
; CHECK: mlg %r2, 524280(%r4)
; CHECK: br %r14
  %ptr = getelementptr i64, i64 *%src, i64 65535
  %b = load i64, i64 *%ptr
  %ax = zext i64 %a to i128
  %bx = zext i64 %b to i128
  %mulx = mul i128 %ax, %bx
  %highx = lshr i128 %mulx, 64
  %high = trunc i128 %highx to i64
  ret i64 %high
}

; Check the next doubleword up, which requires separate address logic.
; Other sequences besides this one would be OK.
define i64 @f8(i64 %dummy, i64 %a, i64 *%src) {
; CHECK-LABEL: f8:
; CHECK: agfi %r4, 524288
; CHECK: mlg %r2, 0(%r4)
; CHECK: br %r14
  %ptr = getelementptr i64, i64 *%src, i64 65536
  %b = load i64, i64 *%ptr
  %ax = zext i64 %a to i128
  %bx = zext i64 %b to i128
  %mulx = mul i128 %ax, %bx
  %highx = lshr i128 %mulx, 64
  %high = trunc i128 %highx to i64
  ret i64 %high
}

; Check the high end of the negative aligned MLG range.
define i64 @f9(i64 %dummy, i64 %a, i64 *%src) {
; CHECK-LABEL: f9:
; CHECK: mlg %r2, -8(%r4)
; CHECK: br %r14
  %ptr = getelementptr i64, i64 *%src, i64 -1
  %b = load i64, i64 *%ptr
  %ax = zext i64 %a to i128
  %bx = zext i64 %b to i128
  %mulx = mul i128 %ax, %bx
  %highx = lshr i128 %mulx, 64
  %high = trunc i128 %highx to i64
  ret i64 %high
}

; Check the low end of the MLG range.
define i64 @f10(i64 %dummy, i64 %a, i64 *%src) {
; CHECK-LABEL: f10:
; CHECK: mlg %r2, -524288(%r4)
; CHECK: br %r14
  %ptr = getelementptr i64, i64 *%src, i64 -65536
  %b = load i64, i64 *%ptr
  %ax = zext i64 %a to i128
  %bx = zext i64 %b to i128
  %mulx = mul i128 %ax, %bx
  %highx = lshr i128 %mulx, 64
  %high = trunc i128 %highx to i64
  ret i64 %high
}

; Check the next doubleword down, which needs separate address logic.
; Other sequences besides this one would be OK.
define i64 @f11(i64 *%dest, i64 %a, i64 *%src) {
; CHECK-LABEL: f11:
; CHECK: agfi %r4, -524296
; CHECK: mlg %r2, 0(%r4)
; CHECK: br %r14
  %ptr = getelementptr i64, i64 *%src, i64 -65537
  %b = load i64, i64 *%ptr
  %ax = zext i64 %a to i128
  %bx = zext i64 %b to i128
  %mulx = mul i128 %ax, %bx
  %highx = lshr i128 %mulx, 64
  %high = trunc i128 %highx to i64
  ret i64 %high
}

; Check that MLG allows an index.
define i64 @f12(i64 *%dest, i64 %a, i64 %src, i64 %index) {
; CHECK-LABEL: f12:
; CHECK: mlg %r2, 524287(%r5,%r4)
; CHECK: br %r14
  %add1 = add i64 %src, %index
  %add2 = add i64 %add1, 524287
  %ptr = inttoptr i64 %add2 to i64 *
  %b = load i64, i64 *%ptr
  %ax = zext i64 %a to i128
  %bx = zext i64 %b to i128
  %mulx = mul i128 %ax, %bx
  %highx = lshr i128 %mulx, 64
  %high = trunc i128 %highx to i64
  ret i64 %high
}

; Check that multiplications of spilled values can use MLG rather than MLGR.
define i64 @f13(i64 *%ptr0) {
; CHECK-LABEL: f13:
; CHECK: brasl %r14, foo@PLT
; CHECK: mlg {{%r[0-9]+}}, 160(%r15)
; CHECK: br %r14
  %ptr1 = getelementptr i64, i64 *%ptr0, i64 2
  %ptr2 = getelementptr i64, i64 *%ptr0, i64 4
  %ptr3 = getelementptr i64, i64 *%ptr0, i64 6
  %ptr4 = getelementptr i64, i64 *%ptr0, i64 8
  %ptr5 = getelementptr i64, i64 *%ptr0, i64 10
  %ptr6 = getelementptr i64, i64 *%ptr0, i64 12
  %ptr7 = getelementptr i64, i64 *%ptr0, i64 14
  %ptr8 = getelementptr i64, i64 *%ptr0, i64 16
  %ptr9 = getelementptr i64, i64 *%ptr0, i64 18

  %val0 = load i64, i64 *%ptr0
  %val1 = load i64, i64 *%ptr1
  %val2 = load i64, i64 *%ptr2
  %val3 = load i64, i64 *%ptr3
  %val4 = load i64, i64 *%ptr4
  %val5 = load i64, i64 *%ptr5
  %val6 = load i64, i64 *%ptr6
  %val7 = load i64, i64 *%ptr7
  %val8 = load i64, i64 *%ptr8
  %val9 = load i64, i64 *%ptr9

  %ret = call i64 @foo()

  %retx = zext i64 %ret to i128
  %val0x = zext i64 %val0 to i128
  %mul0d = mul i128 %retx, %val0x
  %mul0x = lshr i128 %mul0d, 64

  %val1x = zext i64 %val1 to i128
  %mul1d = mul i128 %mul0x, %val1x
  %mul1x = lshr i128 %mul1d, 64

  %val2x = zext i64 %val2 to i128
  %mul2d = mul i128 %mul1x, %val2x
  %mul2x = lshr i128 %mul2d, 64

  %val3x = zext i64 %val3 to i128
  %mul3d = mul i128 %mul2x, %val3x
  %mul3x = lshr i128 %mul3d, 64

  %val4x = zext i64 %val4 to i128
  %mul4d = mul i128 %mul3x, %val4x
  %mul4x = lshr i128 %mul4d, 64

  %val5x = zext i64 %val5 to i128
  %mul5d = mul i128 %mul4x, %val5x
  %mul5x = lshr i128 %mul5d, 64

  %val6x = zext i64 %val6 to i128
  %mul6d = mul i128 %mul5x, %val6x
  %mul6x = lshr i128 %mul6d, 64

  %val7x = zext i64 %val7 to i128
  %mul7d = mul i128 %mul6x, %val7x
  %mul7x = lshr i128 %mul7d, 64

  %val8x = zext i64 %val8 to i128
  %mul8d = mul i128 %mul7x, %val8x
  %mul8x = lshr i128 %mul8d, 64

  %val9x = zext i64 %val9 to i128
  %mul9d = mul i128 %mul8x, %val9x
  %mul9x = lshr i128 %mul9d, 64

  %mul9 = trunc i128 %mul9x to i64
  ret i64 %mul9
}
