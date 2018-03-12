; RUN: llc -march=hexagon < %s | FileCheck %s
; This test checks for the generation of 64b mul instruction
; (dpmpyss_s0 and dpmpyuu_s0).

; Checks for unsigned multiplication.

; 16 x 16 = 64
; CHECK-LABEL: f0:
; CHECK: r1:0 = mpyu(
define i64 @f0(i16 zeroext %a0, i16 zeroext %a1) local_unnamed_addr #0 {
b0:
  %v0 = zext i16 %a0 to i64
  %v1 = zext i16 %a1 to i64
  %v2 = mul nuw nsw i64 %v1, %v0
  ret i64 %v2
}

; 32 x 32 = 64
; CHECK-LABEL: f1:
; CHECK: r1:0 = mpyu(
define i64 @f1(i32 %a0, i32 %a1) local_unnamed_addr #0 {
b0:
  %v0 = zext i32 %a0 to i64
  %v1 = zext i32 %a1 to i64
  %v2 = mul nuw nsw i64 %v1, %v0
  ret i64 %v2
}

; Given int w[2], short h[4], signed char c[8], the below tests check for the
; generation of dpmpyuu_s0.
; w[0] * h[0]
; CHECK-LABEL: f2:
; CHECK: = sxth
; CHECK: r1:0 = mpyu(
define i64 @f2(i64 %a0, i64 %a1) local_unnamed_addr #0 {
b0:
  %v0 = and i64 %a0, 4294967295
  %v1 = trunc i64 %a1 to i32
  %v2 = shl i32 %v1, 16
  %v3 = ashr exact i32 %v2, 16
  %v4 = zext i32 %v3 to i64
  %v5 = mul nuw i64 %v0, %v4
  ret i64 %v5
}

; w[0] * h[1]
; CHECK-LABEL: f3:
; CHECK: = asrh
; CHECK: r1:0 = mpyu(
define i64 @f3(i64 %a0, i64 %a1) local_unnamed_addr #0 {
b0:
  %v0 = and i64 %a0, 4294967295
  %v1 = trunc i64 %a1 to i32
  %v2 = ashr i32 %v1, 16
  %v3 = zext i32 %v2 to i64
  %v4 = mul nuw i64 %v0, %v3
  ret i64 %v4
}

; w[0] * h[2]
; CHECK-LABEL: f4:
; CHECK: = extract(
; CHECK: r1:0 = mpyu(
define i64 @f4(i64 %a0, i64 %a1) local_unnamed_addr #0 {
b0:
  %v0 = and i64 %a0, 4294967295
  %v1 = lshr i64 %a1, 32
  %v2 = shl nuw nsw i64 %v1, 16
  %v3 = trunc i64 %v2 to i32
  %v4 = ashr exact i32 %v3, 16
  %v5 = zext i32 %v4 to i64
  %v6 = mul nuw i64 %v0, %v5
  ret i64 %v6
}

; w[0] * h[3]
; CHECK-LABEL: f5:
; CHECK: = extractu(
; CHECK: r1:0 = mpyu(
define i64 @f5(i64 %a0, i64 %a1) local_unnamed_addr #0 {
b0:
  %v0 = and i64 %a0, 4294967295
  %v1 = lshr i64 %a1, 48
  %v2 = shl nuw nsw i64 %v1, 16
  %v3 = trunc i64 %v2 to i32
  %v4 = ashr exact i32 %v3, 16
  %v5 = zext i32 %v4 to i64
  %v6 = mul nuw i64 %v0, %v5
  ret i64 %v6
}

; w[1] * h[0]
; CHECK-LABEL: f6:
; CHECK: = sxth(
; CHECK: r1:0 = mpyu(
define i64 @f6(i64 %a0, i64 %a1) local_unnamed_addr #0 {
b0:
  %v0 = lshr i64 %a0, 32
  %v1 = trunc i64 %a1 to i32
  %v2 = shl i32 %v1, 16
  %v3 = ashr exact i32 %v2, 16
  %v4 = zext i32 %v3 to i64
  %v5 = mul nuw i64 %v0, %v4
  ret i64 %v5
}

; w[0] * c[0]
; CHECK-LABEL: f7:
; CHECK: = and({{.*}}#255)
; CHECK: r1:0 = mpyu(
define i64 @f7(i64 %a0, i64 %a1) local_unnamed_addr #0 {
b0:
  %v0 = and i64 %a0, 4294967295
  %v1 = and i64 %a1, 255
  %v2 = mul nuw nsw i64 %v1, %v0
  ret i64 %v2
}

; w[0] * c[2]
; CHECK-LABEL: f8:
; CHECK: = extractu(
; CHECK: r1:0 = mpyu(
define i64 @f8(i64 %a0, i64 %a1) local_unnamed_addr #0 {
b0:
  %v0 = and i64 %a0, 4294967295
  %v1 = lshr i64 %a1, 16
  %v2 = and i64 %v1, 255
  %v3 = mul nuw nsw i64 %v2, %v0
  ret i64 %v3
}

; w[0] * c[7]
; CHECK-LABEL: f9:
; CHECK: = lsr(
; CHECK: r1:0 = mpyu(
define i64 @f9(i64 %a0, i64 %a1) local_unnamed_addr #0 {
b0:
  %v0 = and i64 %a0, 4294967295
  %v1 = lshr i64 %a1, 56
  %v2 = mul nuw nsw i64 %v1, %v0
  ret i64 %v2
}


; Checks for signed multiplication.

; 16 x 16 = 64
; CHECK-LABEL: f10:
; CHECK: r1:0 = mpy(
define i64 @f10(i16 signext %a0, i16 signext %a1) local_unnamed_addr #0 {
b0:
  %v0 = sext i16 %a0 to i64
  %v1 = sext i16 %a1 to i64
  %v2 = mul nsw i64 %v1, %v0
  ret i64 %v2
}

; 32 x 32 = 64
; CHECK-LABEL: f11:
; CHECK: r1:0 = mpy(
define i64 @f11(i32 %a0, i32 %a1) local_unnamed_addr #0 {
b0:
  %v0 = sext i32 %a0 to i64
  %v1 = sext i32 %a1 to i64
  %v2 = mul nsw i64 %v1, %v0
  ret i64 %v2
}

; Given unsigned int w[2], unsigned short h[4], unsigned char c[8], the below
; tests check for the generation of dpmpyss_s0.
; w[0] * h[0]
; CHECK-LABEL: f12:
; CHECK: = sxth
; CHECK: r1:0 = mpy(
define i64 @f12(i64 %a0, i64 %a1) local_unnamed_addr #0 {
b0:
  %v0 = shl i64 %a0, 32
  %v1 = ashr exact i64 %v0, 32
  %v2 = shl i64 %a1, 48
  %v3 = ashr exact i64 %v2, 48
  %v4 = mul nsw i64 %v3, %v1
  ret i64 %v4
}

; w[0] * h[1]
; CHECK-LABEL: f13:
; CHECK: = asrh
; CHECK: r1:0 = mpy(
define i64 @f13(i64 %a0, i64 %a1) local_unnamed_addr #0 {
b0:
  %v0 = shl i64 %a0, 32
  %v1 = ashr exact i64 %v0, 32
  %v2 = trunc i64 %a1 to i32
  %v3 = ashr i32 %v2, 16
  %v4 = sext i32 %v3 to i64
  %v5 = mul nsw i64 %v1, %v4
  ret i64 %v5
}

; w[0] * h[2]
; CHECK-LABEL: f14:
; CHECK: = extract(
; CHECK: r1:0 = mpy(
define i64 @f14(i64 %a0, i64 %a1) local_unnamed_addr #0 {
b0:
  %v0 = shl i64 %a0, 32
  %v1 = ashr exact i64 %v0, 32
  %v2 = lshr i64 %a1, 32
  %v3 = shl nuw nsw i64 %v2, 16
  %v4 = trunc i64 %v3 to i32
  %v5 = ashr exact i32 %v4, 16
  %v6 = sext i32 %v5 to i64
  %v7 = mul nsw i64 %v1, %v6
  ret i64 %v7
}

; w[0] * h[3]
; CHECK-LABEL: f15:
; CHECK: = sxth(
; CHECK: r1:0 = mpy(
define i64 @f15(i64 %a0, i64 %a1) local_unnamed_addr #0 {
b0:
  %v0 = ashr i64 %a0, 32
  %v1 = shl i64 %a1, 48
  %v2 = ashr exact i64 %v1, 48
  %v3 = mul nsw i64 %v2, %v0
  ret i64 %v3
}

; w[1] * h[0]
; CHECK-LABEL: f16:
; CHECK: = asrh(
; CHECK: r1:0 = mpy(
define i64 @f16(i64 %a0, i64 %a1) local_unnamed_addr #0 {
b0:
  %v0 = ashr i64 %a0, 32
  %v1 = trunc i64 %a1 to i32
  %v2 = ashr i32 %v1, 16
  %v3 = sext i32 %v2 to i64
  %v4 = mul nsw i64 %v0, %v3
  ret i64 %v4
}

; w[0] * c[0]
; CHECK-LABEL: f17:
; CHECK: = sxtb(
; CHECK: r1:0 = mpy(
define i64 @f17(i64 %a0, i64 %a1) local_unnamed_addr #0 {
b0:
  %v0 = shl i64 %a0, 32
  %v1 = ashr exact i64 %v0, 32
  %v2 = shl i64 %a1, 56
  %v3 = ashr exact i64 %v2, 56
  %v4 = mul nsw i64 %v3, %v1
  ret i64 %v4
}

; w[0] * c[2]
; CHECK-LABEL: f18:
; CHECK: = extract(
; CHECK: r1:0 = mpy(
define i64 @f18(i64 %a0, i64 %a1) local_unnamed_addr #0 {
b0:
  %v0 = shl i64 %a0, 32
  %v1 = ashr exact i64 %v0, 32
  %v2 = lshr i64 %a1, 16
  %v3 = shl i64 %v2, 24
  %v4 = trunc i64 %v3 to i32
  %v5 = ashr exact i32 %v4, 24
  %v6 = sext i32 %v5 to i64
  %v7 = mul nsw i64 %v1, %v6
  ret i64 %v7
}

; w[0] * c[7]
; CHECK-LABEL: f19:
; CHECK: = sxtb(
; CHECK: r1:0 = mpy(
define i64 @f19(i64 %a0, i64 %a1) local_unnamed_addr #0 {
b0:
  %v0 = shl i64 %a0, 32
  %v1 = ashr exact i64 %v0, 32
  %v2 = lshr i64 %a1, 56
  %v3 = shl nuw nsw i64 %v2, 24
  %v4 = trunc i64 %v3 to i32
  %v5 = ashr exact i32 %v4, 24
  %v6 = sext i32 %v5 to i64
  %v7 = mul nsw i64 %v1, %v6
  ret i64 %v7
}

attributes #0 = { norecurse nounwind readnone "target-cpu"="hexagonv60" "target-features"="-hvx" }
