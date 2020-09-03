; RUN: llc < %s -mtriple=arm64 | FileCheck %s

define i1 @testSwapCmpWithLSL64_1(i64 %a, i64 %b) {
; CHECK-LABEL: testSwapCmpWithLSL64_1:
; CHECK:      cmp     x1, x0, lsl #1
; CHECK-NEXT: cset    w0, gt
entry:
  %shl = shl i64 %a, 1
  %cmp = icmp slt i64 %shl, %b
  ret i1 %cmp
}

define i1 @testSwapCmpWithLSL64_63(i64 %a, i64 %b) {
; CHECK-LABEL: testSwapCmpWithLSL64_63:
; CHECK:      cmp     x1, x0, lsl #63
; CHECK-NEXT: cset    w0, gt
entry:
  %shl = shl i64 %a, 63
  %cmp = icmp slt i64 %shl, %b
  ret i1 %cmp
}

define i1 @testSwapCmpWithLSL32_1(i32 %a, i32 %b) {
; CHECK-LABEL: testSwapCmpWithLSL32_1:
; CHECK:      cmp     w1, w0, lsl #1
; CHECK-NEXT: cset    w0, gt
entry:
  %shl = shl i32 %a, 1
  %cmp = icmp slt i32 %shl, %b
  ret i1 %cmp
}

define i1 @testSwapCmpWithLSL32_31(i32 %a, i32 %b) {
; CHECK-LABEL: testSwapCmpWithLSL32_31:
; CHECK:      cmp     w1, w0, lsl #31
; CHECK-NEXT: cset    w0, gt
entry:
  %shl = shl i32 %a, 31
  %cmp = icmp slt i32 %shl, %b
  ret i1 %cmp
}

define i1 @testSwapCmpWithLSR64_1(i64 %a, i64 %b) {
; CHECK-LABEL: testSwapCmpWithLSR64_1:
; CHECK:      cmp     x1, x0, lsr #1
; CHECK-NEXT: cset    w0, gt
entry:
  %lshr = lshr i64 %a, 1
  %cmp = icmp slt i64 %lshr, %b
  ret i1 %cmp
}

define i1 @testSwapCmpWithLSR64_63(i64 %a, i64 %b) {
; CHECK-LABEL: testSwapCmpWithLSR64_63:
; CHECK:      cmp     x1, x0, lsr #63
; CHECK-NEXT: cset    w0, gt
entry:
  %lshr = lshr i64 %a, 63
  %cmp = icmp slt i64 %lshr, %b
  ret i1 %cmp
}

define i1 @testSwapCmpWithLSR32_1(i32 %a, i32 %b) {
; CHECK-LABEL: testSwapCmpWithLSR32_1:
; CHECK:      cmp     w1, w0, lsr #1
; CHECK-NEXT: cset    w0, gt
entry:
  %lshr = lshr i32 %a, 1
  %cmp = icmp slt i32 %lshr, %b
  ret i1 %cmp
}

define i1 @testSwapCmpWithLSR32_31(i32 %a, i32 %b) {
; CHECK-LABEL: testSwapCmpWithLSR32_31:
; CHECK:      cmp     w1, w0, lsr #31
; CHECK-NEXT: cset    w0, gt
entry:
  %lshr = lshr i32 %a, 31
  %cmp = icmp slt i32 %lshr, %b
  ret i1 %cmp
}

define i1 @testSwapCmpWithASR64_1(i64 %a, i64 %b) {
; CHECK-LABEL: testSwapCmpWithASR64_1:
; CHECK:      cmp     x1, x0, asr #1
; CHECK-NEXT: cset    w0, gt
entry:
  %ashr = ashr i64 %a, 1
  %cmp = icmp slt i64 %ashr, %b
  ret i1 %cmp
}

define i1 @testSwapCmpWithASR64_63(i64 %a, i64 %b) {
; CHECK-LABEL: testSwapCmpWithASR64_63:
; CHECK:      cmp     x1, x0, asr #63
; CHECK-NEXT: cset    w0, gt
entry:
  %ashr = ashr i64 %a, 63
  %cmp = icmp slt i64 %ashr, %b
  ret i1 %cmp
}

define i1 @testSwapCmpWithASR32_1(i32 %a, i32 %b) {
; CHECK-LABEL: testSwapCmpWithASR32_1:
; CHECK:      cmp     w1, w0, asr #1
; CHECK-NEXT: cset    w0, gt
entry:
  %ashr = ashr i32 %a, 1
  %cmp = icmp slt i32 %ashr, %b
  ret i1 %cmp
}

define i1 @testSwapCmpWithASR32_31(i32 %a, i32 %b) {
; CHECK-LABEL: testSwapCmpWithASR32_31:
; CHECK:      cmp     w1, w0, asr #31
; CHECK-NEXT: cset    w0, gt
entry:
  %ashr = ashr i32 %a, 31
  %cmp = icmp slt i32 %ashr, %b
  ret i1 %cmp
}

define i1 @testSwapCmpWithShiftedZeroExtend32_64(i32 %a, i64 %b) {
; CHECK-LABEL: testSwapCmpWithShiftedZeroExtend32_64
; CHECK:      cmp    x1, w0, uxtw #2
; CHECK-NEXT: cset   w0, lo
entry:
  %a64 = zext i32 %a to i64
  %shl.0 = shl i64 %a64, 2
  %cmp = icmp ugt i64 %shl.0, %b
  ret i1 %cmp
}

define i1 @testSwapCmpWithShiftedZeroExtend16_64(i16 %a, i64 %b) {
; CHECK-LABEL: testSwapCmpWithShiftedZeroExtend16_64
; CHECK:      cmp    x1, w0, uxth #2
; CHECK-NEXT: cset   w0, lo
entry:
  %a64 = zext i16 %a to i64
  %shl.0 = shl i64 %a64, 2
  %cmp = icmp ugt i64 %shl.0, %b
  ret i1 %cmp
}

define i1 @testSwapCmpWithShiftedZeroExtend8_64(i8 %a, i64 %b) {
; CHECK-LABEL: testSwapCmpWithShiftedZeroExtend8_64
; CHECK:      cmp    x1, w0, uxtb #4
; CHECK-NEXT: cset    w0, lo
entry:
  %a64 = zext i8 %a to i64
  %shl.2 = shl i64 %a64, 4
  %cmp = icmp ugt i64 %shl.2, %b
  ret i1 %cmp
}

define i1 @testSwapCmpWithShiftedZeroExtend16_32(i16 %a, i32 %b) {
; CHECK-LABEL: testSwapCmpWithShiftedZeroExtend16_32
; CHECK:      cmp    w1, w0, uxth #3
; CHECK-NEXT: cset    w0, lo
entry:
  %a32 = zext i16 %a to i32
  %shl = shl i32 %a32, 3
  %cmp = icmp ugt i32 %shl, %b
  ret i1 %cmp
}

define i1 @testSwapCmpWithShiftedZeroExtend8_32(i8 %a, i32 %b) {
; CHECK-LABEL: testSwapCmpWithShiftedZeroExtend8_32
; CHECK:      cmp    w1, w0, uxtb #4
; CHECK-NEXT: cset    w0, lo
entry:
  %a32 = zext i8 %a to i32
  %shl = shl i32 %a32, 4
  %cmp = icmp ugt i32 %shl, %b
  ret i1 %cmp
}

define i1 @testSwapCmpWithTooLargeShiftedZeroExtend8_32(i8 %a, i32 %b) {
; CHECK-LABEL: testSwapCmpWithTooLargeShiftedZeroExtend8_32
; CHECK:      and    [[REG:w[0-9]+]], w0, #0xff
; CHECK:      cmp    w1, [[REG]], lsl #5
; CHECK-NEXT: cset   w0, lo
entry:
  %a32 = zext i8 %a to i32
  %shl = shl i32 %a32, 5
  %cmp = icmp ugt i32 %shl, %b
  ret i1 %cmp
}

define i1 @testSwapCmpWithZeroExtend8_32(i8 %a, i32 %b) {
; CHECK-LABEL: testSwapCmpWithZeroExtend8_32
; CHECK:      cmp    w1, w0, uxtb
; CHECK-NEXT: cset   w0, lo
entry:
  %a32 = zext i8 %a to i32
  %cmp = icmp ugt i32 %a32, %b
  ret i1 %cmp
}

define i1 @testSwapCmpWithShiftedSignExtend32_64(i32 %a, i64 %b) {
; CHECK-LABEL: testSwapCmpWithShiftedSignExtend32_64
; CHECK:      cmp    x1, w0, sxtw #2
; CHECK-NEXT: cset   w0, lo
entry:
  %a64 = sext i32 %a to i64
  %shl.0 = shl i64 %a64, 2
  %cmp = icmp ugt i64 %shl.0, %b
  ret i1 %cmp
}

define i1 @testSwapCmpWithShiftedSignExtend16_64(i16 %a, i64 %b) {
; CHECK-LABEL: testSwapCmpWithShiftedSignExtend16_64
; CHECK:      cmp    x1, w0, sxth #2
; CHECK-NEXT: cset   w0, lo
entry:
  %a64 = sext i16 %a to i64
  %shl.0 = shl i64 %a64, 2
  %cmp = icmp ugt i64 %shl.0, %b
  ret i1 %cmp
}

define i1 @testSwapCmpWithShiftedSignExtend8_64(i8 %a, i64 %b) {
; CHECK-LABEL: testSwapCmpWithShiftedSignExtend8_64
; CHECK:      cmp    x1, w0, sxtb #4
; CHECK-NEXT: cset    w0, lo
entry:
  %a64 = sext i8 %a to i64
  %shl.2 = shl i64 %a64, 4
  %cmp = icmp ugt i64 %shl.2, %b
  ret i1 %cmp
}

define i1 @testSwapCmpWithShiftedSignExtend16_32(i16 %a, i32 %b) {
; CHECK-LABEL: testSwapCmpWithShiftedSignExtend16_32
; CHECK:      cmp    w1, w0, sxth #3
; CHECK-NEXT: cset    w0, lo
entry:
  %a32 = sext i16 %a to i32
  %shl = shl i32 %a32, 3
  %cmp = icmp ugt i32 %shl, %b
  ret i1 %cmp
}

define i1 @testSwapCmpWithShiftedSignExtend8_32(i8 %a, i32 %b) {
; CHECK-LABEL: testSwapCmpWithShiftedSignExtend8_32
; CHECK:      cmp    w1, w0, sxtb #4
; CHECK-NEXT: cset   w0, lo
entry:
  %a32 = sext i8 %a to i32
  %shl = shl i32 %a32, 4
  %cmp = icmp ugt i32 %shl, %b
  ret i1 %cmp
}

define i1 @testSwapCmpWithTooLargeShiftedSignExtend8_32(i8 %a, i32 %b) {
; CHECK-LABEL: testSwapCmpWithTooLargeShiftedSignExtend8_32
; CHECK:      sxtb   [[REG:w[0-9]+]], w0
; CHECK-NEXT: cmp    w1, [[REG]], lsl #5
; CHECK-NEXT: cset   w0, lo
entry:
  %a32 = sext i8 %a to i32
  %shl = shl i32 %a32, 5
  %cmp = icmp ugt i32 %shl, %b
  ret i1 %cmp
}

define i1 @testSwapCmpWithSignExtend8_32(i8 %a, i32 %b) {
; CHECK-LABEL: testSwapCmpWithSignExtend8_32
; CHECK:      cmp    w1, w0, sxtb
; CHECK-NEXT: cset   w0, lo
entry:
  %a32 = sext i8 %a to i32
  %cmp = icmp ugt i32 %a32, %b
  ret i1 %cmp
}

define i1 @testSwapCmnWithLSL64_1(i64 %a, i64 %b) {
; CHECK-LABEL: testSwapCmnWithLSL64_1:
; CHECK:      cmn    x1, x0, lsl #1
; CHECK-NEXT: cset   w0, ne
entry:
  %shl = shl i64 %a, 1
  %na = sub i64 0, %shl
  %cmp = icmp ne i64 %na, %b
  ret i1 %cmp
}

; Note: testing with a 62 bits shift as 63 has another optimization kicking in.
define i1 @testSwapCmnWithLSL64_62(i64 %a, i64 %b) {
; CHECK-LABEL: testSwapCmnWithLSL64_62:
; CHECK:      cmn    x1, x0, lsl #62
; CHECK-NEXT: cset   w0, ne
entry:
  %shl = shl i64 %a, 62
  %na = sub i64 0, %shl
  %cmp = icmp ne i64 %na, %b
  ret i1 %cmp
}

; Note: the 63 bits shift triggers a different optimization path, which leads
; to a similar result in terms of performances. We try to catch here any change
; so that this test can be adapted should the optimization be done with the
; operand swap.
define i1 @testSwapCmnWithLSL64_63(i64 %a, i64 %b) {
; CHECK-LABEL: testSwapCmnWithLSL64_63:
; CHECK:      cmp    x1, x0, lsl #63
; CHECK-NEXT: cset   w0, ne
entry:
  %shl = shl i64 %a, 63
  %na = sub i64 0, %shl
  %cmp = icmp ne i64 %na, %b
  ret i1 %cmp
}

define i1 @testSwapCmnWithLSL32_1(i32 %a, i32 %b) {
; CHECK-LABEL: testSwapCmnWithLSL32_1:
; CHECK:      cmn    w1, w0, lsl #1
; CHECK-NEXT: cset   w0, ne
entry:
  %shl = shl i32 %a, 1
  %na = sub i32 0, %shl
  %cmp = icmp ne i32 %na, %b
  ret i1 %cmp
}

; Note: testing with a 30 bits shift as 30 has another optimization kicking in.
define i1 @testSwapCmnWithLSL32_30(i32 %a, i32 %b) {
; CHECK-LABEL: testSwapCmnWithLSL32_30:
; CHECK:      cmn    w1, w0, lsl #30
; CHECK-NEXT: cset   w0, ne
entry:
  %shl = shl i32 %a, 30
  %na = sub i32 0, %shl
  %cmp = icmp ne i32 %na, %b
  ret i1 %cmp
}

; Note: the 31 bits shift triggers a different optimization path, which leads
; to a similar result in terms of performances. We try to catch here any change
; so that this test can be adapted should the optimization be done with the
; operand swap.
define i1 @testSwapCmnWithLSL32_31(i32 %a, i32 %b) {
; CHECK-LABEL: testSwapCmnWithLSL32_31:
; CHECK:      cmp    w1, w0, lsl #31
; CHECK-NEXT: cset   w0, ne
entry:
  %shl = shl i32 %a, 31
  %na = sub i32 0, %shl
  %cmp = icmp ne i32 %na, %b
  ret i1 %cmp
}

define i1 @testSwapCmnWithLSR64_1(i64 %a, i64 %b) {
; CHECK-LABEL: testSwapCmnWithLSR64_1:
; CHECK:      cmn    x1, x0, lsr #1
; CHECK-NEXT: cset   w0, ne
entry:
  %lshr = lshr i64 %a, 1
  %na = sub i64 0, %lshr
  %cmp = icmp ne i64 %na, %b
  ret i1 %cmp
}

; Note: testing with a 62 bits shift as 63 has another optimization kicking in.
define i1 @testSwapCmnWithLSR64_62(i64 %a, i64 %b) {
; CHECK-LABEL: testSwapCmnWithLSR64_62:
; CHECK:      cmn    x1, x0, lsr #62
; CHECK-NEXT: cset   w0, ne
entry:
  %lshr = lshr i64 %a, 62
  %na = sub i64 0, %lshr
  %cmp = icmp ne i64 %na, %b
  ret i1 %cmp
}

; Note: the 63 bits shift triggers a different optimization path, which leads
; to a similar result in terms of performances. We try to catch here any change
; so that this test can be adapted should the optimization be done with the
; operand swap.
define i1 @testSwapCmnWithLSR64_63(i64 %a, i64 %b) {
; CHECK-LABEL: testSwapCmnWithLSR64_63:
; CHECK:      cmp    x1, x0, asr #63
; CHECK-NEXT: cset   w0, ne
entry:
  %lshr = lshr i64 %a, 63
  %na = sub i64 0, %lshr
  %cmp = icmp ne i64 %na, %b
  ret i1 %cmp
}

define i1 @testSwapCmnWithLSR32_1(i32 %a, i32 %b) {
; CHECK-LABEL: testSwapCmnWithLSR32_1:
; CHECK:      cmn    w1, w0, lsr #1
; CHECK-NEXT: cset   w0, ne
entry:
  %lshr = lshr i32 %a, 1
  %na = sub i32 0, %lshr
  %cmp = icmp ne i32 %na, %b
  ret i1 %cmp
}

; Note: testing with a 30 bits shift as 31 has another optimization kicking in.
define i1 @testSwapCmnWithLSR32_30(i32 %a, i32 %b) {
; CHECK-LABEL: testSwapCmnWithLSR32_30:
; CHECK:      cmn    w1, w0, lsr #30
; CHECK-NEXT: cset   w0, ne
entry:
  %lshr = lshr i32 %a, 30
  %na = sub i32 0, %lshr
  %cmp = icmp ne i32 %na, %b
  ret i1 %cmp
}

; Note: the 31 bits shift triggers a different optimization path, which leads
; to a similar result in terms of performances. We try to catch here any change
; so that this test can be adapted should the optimization be done with the
; operand swap.
define i1 @testSwapCmnWithLSR32_31(i32 %a, i32 %b) {
; CHECK-LABEL: testSwapCmnWithLSR32_31:
; CHECK:      cmp    w1, w0, asr #31
; CHECK-NEXT: cset   w0, ne
entry:
  %lshr = lshr i32 %a, 31
  %na = sub i32 0, %lshr
  %cmp = icmp ne i32 %na, %b
  ret i1 %cmp
}

define i1 @testSwapCmnWithASR64_1(i64 %a, i64 %b) {
; CHECK-LABEL: testSwapCmnWithASR64_1:
; CHECK:      cmn    x1, x0, asr #3
; CHECK-NEXT: cset   w0, ne
entry:
  %lshr = ashr i64 %a, 3
  %na = sub i64 0, %lshr
  %cmp = icmp ne i64 %na, %b
  ret i1 %cmp
}

; Note: testing with a 62 bits shift as 63 has another optimization kicking in.
define i1 @testSwapCmnWithASR64_62(i64 %a, i64 %b) {
; CHECK-LABEL: testSwapCmnWithASR64_62:
; CHECK:      cmn    x1, x0, asr #62
; CHECK-NEXT: cset   w0, ne
entry:
  %lshr = ashr i64 %a, 62
  %na = sub i64 0, %lshr
  %cmp = icmp ne i64 %na, %b
  ret i1 %cmp
}

; Note: the 63 bits shift triggers a different optimization path, which leads
; to a similar result in terms of performances. We try to catch here any change
; so that this test can be adapted should the optimization be done with the
; operand swap.
define i1 @testSwapCmnWithASR64_63(i64 %a, i64 %b) {
; CHECK-LABEL: testSwapCmnWithASR64_63:
; CHECK:      cmp    x1, x0, lsr #63
; CHECK-NEXT: cset   w0, ne
entry:
  %lshr = ashr i64 %a, 63
  %na = sub i64 0, %lshr
  %cmp = icmp ne i64 %na, %b
  ret i1 %cmp
}

define i1 @testSwapCmnWithASR32_1(i32 %a, i32 %b) {
; CHECK-LABEL: testSwapCmnWithASR32_1:
; CHECK:      cmn    w1, w0, asr #1
; CHECK-NEXT: cset   w0, eq
entry:
  %lshr = ashr i32 %a, 1
  %na = sub i32 0, %lshr
  %cmp = icmp eq i32 %na, %b
  ret i1 %cmp
}

; Note: testing with a 30 bits shift as 31 has another optimization kicking in.
define i1 @testSwapCmnWithASR32_30(i32 %a, i32 %b) {
; CHECK-LABEL: testSwapCmnWithASR32_30:
; CHECK:      cmn    w1, w0, asr #30
; CHECK-NEXT: cset   w0, ne
entry:
  %lshr = ashr i32 %a, 30
  %na = sub i32 0, %lshr
  %cmp = icmp ne i32 %na, %b
  ret i1 %cmp
}

; Note: the 31 bits shift triggers a different optimization path, which leads
; to a similar result in terms of performances. We try to catch here any change
; so that this test can be adapted should the optimization be done with the
; operand swap.
define i1 @testSwapCmnWithASR32_31(i32 %a, i32 %b) {
; CHECK-LABEL: testSwapCmnWithASR32_31:
; CHECK:      cmp    w1, w0, lsr #31
; CHECK-NEXT: cset   w0, ne
entry:
  %lshr = ashr i32 %a, 31
  %na = sub i32 0, %lshr
  %cmp = icmp ne i32 %na, %b
  ret i1 %cmp
}

define i64 @testSwapCmpToCmnWithZeroExtend(i32 %a32, i16 %a16, i8 %a8, i64 %b64, i32 %b32) {
; CHECK-LABEL: testSwapCmpToCmnWithZeroExtend:
t0:
  %conv0 = zext i32 %a32 to i64
  %shl0 = shl i64 %conv0, 1
  %na0 = sub i64 0, %shl0
  %cmp0 = icmp ne i64 %na0, %b64
; CHECK: cmn    x3, w0, uxtw #1
  br i1 %cmp0, label %t1, label %end

t1:
  %conv1 = zext i16 %a16 to i64
  %shl1 = shl i64 %conv1, 4
  %na1 = sub i64 0, %shl1
  %cmp1 = icmp ne i64 %na1, %b64
; CHECK: cmn    x3, w1, uxth #4
  br i1 %cmp1, label %t2, label %end

t2:
  %conv2 = zext i8 %a8 to i64
  %shl2 = shl i64 %conv2, 3
  %na2 = sub i64 0, %shl2
  %cmp2 = icmp ne i64 %na2, %b64
; CHECK: cmn    x3, w2, uxtb #3
  br i1 %cmp2, label %t3, label %end

t3:
  %conv3 = zext i16 %a16 to i32
  %shl3 = shl i32 %conv3, 2
  %na3 = sub i32 0, %shl3
  %cmp3 = icmp ne i32 %na3, %b32
; CHECK: cmn    w4, w1, uxth #2
  br i1 %cmp3, label %t4, label %end

t4:
  %conv4 = zext i8 %a8 to i32
  %shl4 = shl i32 %conv4, 1
  %na4 = sub i32 0, %shl4
  %cmp4 = icmp ne i32 %na4, %b32
; CHECK: cmn    w4, w2, uxtb #1
  br i1 %cmp4, label %t5, label %end

t5:
  %conv5 = zext i8 %a8 to i32
  %shl5 = shl i32 %conv5, 5
  %na5 = sub i32 0, %shl5
  %cmp5 = icmp ne i32 %na5, %b32
; CHECK: and    [[REG:w[0-9]+]], w2, #0xff
; CHECK: cmn    w4, [[REG]], lsl #5
  br i1 %cmp5, label %t6, label %end

t6:
  %conv6 = zext i8 %a8 to i32
  %na6 = sub i32 0, %conv6
  %cmp6 = icmp ne i32 %na6, %b32
; CHECK: cmn    w4, w2, uxtb
  br i1 %cmp6, label %t7, label %end

t7:
  ret i64 0

end:
  ret i64 1
}
define i64 @testSwapCmpToCmnWithSignExtend(i32 %a32, i16 %a16, i8 %a8, i64 %b64, i32 %b32) {
; CHECK-LABEL: testSwapCmpToCmnWithSignExtend:
t0:
  %conv0 = sext i32 %a32 to i64
  %shl0 = shl i64 %conv0, 1
  %na0 = sub i64 0, %shl0
  %cmp0 = icmp ne i64 %na0, %b64
; CHECK: cmn     x3, w0, sxtw #1
  br i1 %cmp0, label %t1, label %end

t1:
  %conv1 = sext i16 %a16 to i64
  %shl1 = shl i64 %conv1, 4
  %na1 = sub i64 0, %shl1
  %cmp1 = icmp ne i64 %na1, %b64
; CHECK: cmn     x3, w1, sxth #4
  br i1 %cmp1, label %t2, label %end

t2:
  %conv2 = sext i8 %a8 to i64
  %shl2 = shl i64 %conv2, 3
  %na2 = sub i64 0, %shl2
  %cmp2 = icmp ne i64 %na2, %b64
; CHECK: cmn     x3, w2, sxtb #3
  br i1 %cmp2, label %t3, label %end

t3:
  %conv3 = sext i16 %a16 to i32
  %shl3 = shl i32 %conv3, 2
  %na3 = sub i32 0, %shl3
  %cmp3 = icmp ne i32 %na3, %b32
; CHECK: cmn     w4, w1, sxth #2
  br i1 %cmp3, label %t4, label %end

t4:
  %conv4 = sext i8 %a8 to i32
  %shl4 = shl i32 %conv4, 1
  %na4 = sub i32 0, %shl4
  %cmp4 = icmp ne i32 %na4, %b32
; CHECK: cmn     w4, w2, sxtb #1
  br i1 %cmp4, label %t5, label %end

t5:
  %conv5 = sext i8 %a8 to i32
  %shl5 = shl i32 %conv5, 5
  %na5 = sub i32 0, %shl5
  %cmp5 = icmp ne i32 %na5, %b32
; CHECK: sxtb    [[REG:w[0-9]+]], w2
; CHECK: cmn     w4, [[REG]], lsl #5
  br i1 %cmp5, label %t6, label %end

t6:
  %conv6 = sext i8 %a8 to i32
  %na6 = sub i32 0, %conv6
  %cmp6 = icmp ne i32 %na6, %b32
; CHECK: cmn     w4, w2, sxtb
  br i1 %cmp6, label %t7, label %end

t7:
  ret i64 0

end:
  ret i64 1
}
