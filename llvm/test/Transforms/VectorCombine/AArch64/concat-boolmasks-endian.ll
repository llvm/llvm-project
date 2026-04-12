; RUN: opt -passes='vector-combine' -S -mtriple=aarch64-unknown-linux-gnu %s -o - | FileCheck %s --check-prefixes=CHECK,LE
; RUN: opt -passes='vector-combine' -S -mtriple=aarch64_be-unknown-linux-gnu %s -o - | FileCheck %s --check-prefixes=CHECK,BE

define i32 @movmsk_i32_v8i32_v4i32(<4 x i32> %v0, <4 x i32> %v1) {
; CHECK-LABEL: define i32 @movmsk_i32_v8i32_v4i32(
; CHECK-SAME: <4 x i32> [[V0:%.*]], <4 x i32> [[V1:%.*]]) {
; LE-NEXT:    [[TMP1:%.*]] = shufflevector <4 x i32> [[V1]], <4 x i32> [[V0]], <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
; BE-NEXT:    [[TMP1:%.*]] = shufflevector <4 x i32> [[V0]], <4 x i32> [[V1]], <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
; CHECK-NEXT:    [[TMP2:%.*]] = icmp slt <8 x i32> [[TMP1]], zeroinitializer
; CHECK-NEXT:    [[TMP3:%.*]] = bitcast <8 x i1> [[TMP2]] to i8
; CHECK-NEXT:    [[OR:%.*]] = zext i8 [[TMP3]] to i32
; CHECK-NEXT:    ret i32 [[OR]]
;
  %c0 = icmp slt <4 x i32> %v0, zeroinitializer
  %c1 = icmp slt <4 x i32> %v1, zeroinitializer
  %b0 = bitcast <4 x i1> %c0 to i4
  %b1 = bitcast <4 x i1> %c1 to i4
  %z0 = zext i4 %b0 to i32
  %z1 = zext i4 %b1 to i32
  %s0 = shl nuw i32 %z0, 4
  %or = or disjoint i32 %s0, %z1
  ret i32 %or
}
