; RUN: llc < %s -mtriple=ve-unknown-unknown | FileCheck %s

define signext i8 @func8s(i8 signext %0, i8 signext %1) {
; CHECK-LABEL: func8s:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    adds.w.sx %s0, %s1, %s0
; CHECK-NEXT:    sll %s0, %s0, 56
; CHECK-NEXT:    sra.l %s0, %s0, 56
; CHECK-NEXT:    or %s11, 0, %s9
  %3 = add i8 %1, %0
  ret i8 %3
}

define signext i16 @func16s(i16 signext %0, i16 signext %1) {
; CHECK-LABEL: func16s:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    adds.w.sx %s0, %s1, %s0
; CHECK-NEXT:    sll %s0, %s0, 48
; CHECK-NEXT:    sra.l %s0, %s0, 48
; CHECK-NEXT:    or %s11, 0, %s9
  %3 = add i16 %1, %0
  ret i16 %3
}

define signext i32 @func32s(i32 signext %0, i32 signext %1) {
; CHECK-LABEL: func32s:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    adds.w.sx %s0, %s1, %s0
; CHECK-NEXT:    adds.w.sx %s0, %s0, (0)1
; CHECK-NEXT:    or %s11, 0, %s9
  %3 = add nsw i32 %1, %0
  ret i32 %3
}

define i64 @func64s(i64 %0, i64 %1) {
; CHECK-LABEL: func64s:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    adds.l %s0, %s1, %s0
; CHECK-NEXT:    or %s11, 0, %s9
  %3 = add nsw i64 %1, %0
  ret i64 %3
}

define i128 @func128s(i128 %0, i128 %1) {
; CHECK-LABEL: func128s:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    adds.l %s1, %s3, %s1
; CHECK-NEXT:    adds.l %s0, %s2, %s0
; CHECK-NEXT:    cmpu.l %s2, %s0, %s2
; CHECK-NEXT:    or %s3, 0, (0)1
; CHECK-NEXT:    cmov.l.lt %s3, (63)0, %s2
; CHECK-NEXT:    adds.w.zx %s2, %s3, (0)1
; CHECK-NEXT:    adds.l %s1, %s1, %s2
; CHECK-NEXT:    or %s11, 0, %s9
  %3 = add nsw i128 %1, %0
  ret i128 %3
}

define zeroext i8 @func8z(i8 zeroext %0, i8 zeroext %1) {
; CHECK-LABEL: func8z:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    adds.w.sx %s0, %s1, %s0
; CHECK-NEXT:    and %s0, %s0, (56)0
; CHECK-NEXT:    or %s11, 0, %s9
  %3 = add i8 %1, %0
  ret i8 %3
}

define zeroext i16 @func16z(i16 zeroext %0, i16 zeroext %1) {
; CHECK-LABEL: func16z:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    adds.w.sx %s0, %s1, %s0
; CHECK-NEXT:    and %s0, %s0, (48)0
; CHECK-NEXT:    or %s11, 0, %s9
  %3 = add i16 %1, %0
  ret i16 %3
}

define zeroext i32 @func32z(i32 zeroext %0, i32 zeroext %1) {
; CHECK-LABEL: func32z:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    adds.w.sx %s0, %s1, %s0
; CHECK-NEXT:    adds.w.zx %s0, %s0, (0)1
; CHECK-NEXT:    or %s11, 0, %s9
  %3 = add i32 %1, %0
  ret i32 %3
}

define i64 @func64z(i64 %0, i64 %1) {
; CHECK-LABEL: func64z:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    adds.l %s0, %s1, %s0
; CHECK-NEXT:    or %s11, 0, %s9
  %3 = add i64 %1, %0
  ret i64 %3
}

define i128 @func128z(i128 %0, i128 %1) {
; CHECK-LABEL: func128z:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    adds.l %s1, %s3, %s1
; CHECK-NEXT:    adds.l %s0, %s2, %s0
; CHECK-NEXT:    cmpu.l %s2, %s0, %s2
; CHECK-NEXT:    or %s3, 0, (0)1
; CHECK-NEXT:    cmov.l.lt %s3, (63)0, %s2
; CHECK-NEXT:    adds.w.zx %s2, %s3, (0)1
; CHECK-NEXT:    adds.l %s1, %s1, %s2
; CHECK-NEXT:    or %s11, 0, %s9
  %3 = add i128 %1, %0
  ret i128 %3
}

define signext i8 @funci8s(i8 signext %0) {
; CHECK-LABEL: funci8s:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    adds.w.sx %s0, 5, %s0
; CHECK-NEXT:    sll %s0, %s0, 56
; CHECK-NEXT:    sra.l %s0, %s0, 56
; CHECK-NEXT:    or %s11, 0, %s9
  %2 = add i8 %0, 5
  ret i8 %2
}

define signext i16 @funci16s(i16 signext %0) {
; CHECK-LABEL: funci16s:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    adds.w.sx %s0, 5, %s0
; CHECK-NEXT:    sll %s0, %s0, 48
; CHECK-NEXT:    sra.l %s0, %s0, 48
; CHECK-NEXT:    or %s11, 0, %s9
  %2 = add i16 %0, 5
  ret i16 %2
}

define signext i32 @funci32s(i32 signext %0) {
; CHECK-LABEL: funci32s:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    adds.w.sx %s0, 5, %s0
; CHECK-NEXT:    adds.w.sx %s0, %s0, (0)1
; CHECK-NEXT:    or %s11, 0, %s9
  %2 = add nsw i32 %0, 5
  ret i32 %2
}

define i64 @funci64s(i64 %0) {
; CHECK-LABEL: funci64s:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    lea %s0, 5(, %s0)
; CHECK-NEXT:    or %s11, 0, %s9
  %2 = add nsw i64 %0, 5
  ret i64 %2
}

define i128 @funci128s(i128 %0) {
; CHECK-LABEL: funci128s:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    lea %s2, 5(, %s0)
; CHECK-NEXT:    cmpu.l %s0, %s2, %s0
; CHECK-NEXT:    or %s3, 0, (0)1
; CHECK-NEXT:    cmov.l.lt %s3, (63)0, %s0
; CHECK-NEXT:    adds.w.zx %s0, %s3, (0)1
; CHECK-NEXT:    adds.l %s1, %s1, %s0
; CHECK-NEXT:    or %s0, 0, %s2
; CHECK-NEXT:    or %s11, 0, %s9
  %2 = add nsw i128 %0, 5
  ret i128 %2
}

define zeroext i8 @funci8z(i8 zeroext %0) {
; CHECK-LABEL: funci8z:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    adds.w.sx %s0, 5, %s0
; CHECK-NEXT:    and %s0, %s0, (56)0
; CHECK-NEXT:    or %s11, 0, %s9
  %2 = add i8 %0, 5
  ret i8 %2
}

define zeroext i16 @funci16z(i16 zeroext %0) {
; CHECK-LABEL: funci16z:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    adds.w.sx %s0, 5, %s0
; CHECK-NEXT:    and %s0, %s0, (48)0
; CHECK-NEXT:    or %s11, 0, %s9
  %2 = add i16 %0, 5
  ret i16 %2
}

define zeroext i32 @funci32z(i32 zeroext %0) {
; CHECK-LABEL: funci32z:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    adds.w.sx %s0, 5, %s0
; CHECK-NEXT:    adds.w.zx %s0, %s0, (0)1
; CHECK-NEXT:    or %s11, 0, %s9
  %2 = add i32 %0, 5
  ret i32 %2
}

define i64 @funci64z(i64 %0) {
; CHECK-LABEL: funci64z:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    lea %s0, 5(, %s0)
; CHECK-NEXT:    or %s11, 0, %s9
  %2 = add i64 %0, 5
  ret i64 %2
}

define i128 @funci128z(i128 %0) {
; CHECK-LABEL: funci128z:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    lea %s2, 5(, %s0)
; CHECK-NEXT:    cmpu.l %s0, %s2, %s0
; CHECK-NEXT:    or %s3, 0, (0)1
; CHECK-NEXT:    cmov.l.lt %s3, (63)0, %s0
; CHECK-NEXT:    adds.w.zx %s0, %s3, (0)1
; CHECK-NEXT:    adds.l %s1, %s1, %s0
; CHECK-NEXT:    or %s0, 0, %s2
; CHECK-NEXT:    or %s11, 0, %s9
  %2 = add i128 %0, 5
  ret i128 %2
}

define i64 @funci64_2(i64 %0) {
; CHECK-LABEL: funci64_2:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    lea %s1, -2147483648
; CHECK-NEXT:    and %s1, %s1, (32)0
; CHECK-NEXT:    adds.l %s0, %s0, %s1
; CHECK-NEXT:    or %s11, 0, %s9
  %2 = add nsw i64 %0, 2147483648
  ret i64 %2
}

define i128 @funci128_2(i128 %0) {
; CHECK-LABEL: funci128_2:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    lea %s2, -2147483648
; CHECK-NEXT:    and %s2, %s2, (32)0
; CHECK-NEXT:    adds.l %s2, %s0, %s2
; CHECK-NEXT:    cmpu.l %s0, %s2, %s0
; CHECK-NEXT:    or %s3, 0, (0)1
; CHECK-NEXT:    cmov.l.lt %s3, (63)0, %s0
; CHECK-NEXT:    adds.w.zx %s0, %s3, (0)1
; CHECK-NEXT:    adds.l %s1, %s1, %s0
; CHECK-NEXT:    or %s0, 0, %s2
; CHECK-NEXT:    or %s11, 0, %s9
  %2 = add nsw i128 %0, 2147483648
  ret i128 %2
}
