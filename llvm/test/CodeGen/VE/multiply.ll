; RUN: llc < %s -mtriple=ve-unknown-unknown | FileCheck %s

define signext i8 @func8s(i8 signext %a, i8 signext %b) {
; CHECK-LABEL: func8s:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    muls.w.sx %s0, %s1, %s0
; CHECK-NEXT:    sll %s0, %s0, 56
; CHECK-NEXT:    sra.l %s0, %s0, 56
; CHECK-NEXT:    or %s11, 0, %s9
  %r = mul i8 %b, %a
  ret i8 %r
}

define signext i16 @func16s(i16 signext %a, i16 signext %b) {
; CHECK-LABEL: func16s:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    muls.w.sx %s0, %s1, %s0
; CHECK-NEXT:    sll %s0, %s0, 48
; CHECK-NEXT:    sra.l %s0, %s0, 48
; CHECK-NEXT:    or %s11, 0, %s9
  %r = mul i16 %b, %a
  ret i16 %r
}

define signext i32 @func32s(i32 signext %a, i32 signext %b) {
; CHECK-LABEL: func32s:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    muls.w.sx %s0, %s1, %s0
; CHECK-NEXT:    adds.w.sx %s0, %s0, (0)1
; CHECK-NEXT:    or %s11, 0, %s9
  %r = mul nsw i32 %b, %a
  ret i32 %r
}

define i64 @func64(i64 %a, i64 %b) {
; CHECK-LABEL: func64:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    muls.l %s0, %s1, %s0
; CHECK-NEXT:    or %s11, 0, %s9
  %r = mul nsw i64 %b, %a
  ret i64 %r
}

define i128 @func128(i128 %a, i128 %b) {
; CHECK-LABEL: func128:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    or %s4, 0, %s1
; CHECK-NEXT:    or %s5, 0, %s0
; CHECK-NEXT:    lea %s0, __multi3@lo
; CHECK-NEXT:    and %s0, %s0, (32)0
; CHECK-NEXT:    lea.sl %s12, __multi3@hi(, %s0)
; CHECK-NEXT:    or %s0, 0, %s2
; CHECK-NEXT:    or %s1, 0, %s3
; CHECK-NEXT:    or %s2, 0, %s5
; CHECK-NEXT:    or %s3, 0, %s4
; CHECK-NEXT:    bsic %s10, (, %s12)
; CHECK-NEXT:    or %s11, 0, %s9
  %r = mul nsw i128 %b, %a
  ret i128 %r
}

define zeroext i8 @func8z(i8 zeroext %a, i8 zeroext %b) {
; CHECK-LABEL: func8z:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    muls.w.sx %s0, %s1, %s0
; CHECK-NEXT:    and %s0, %s0, (56)0
; CHECK-NEXT:    or %s11, 0, %s9
  %r = mul i8 %b, %a
  ret i8 %r
}

define zeroext i16 @func16z(i16 zeroext %a, i16 zeroext %b) {
; CHECK-LABEL: func16z:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    muls.w.sx %s0, %s1, %s0
; CHECK-NEXT:    and %s0, %s0, (48)0
; CHECK-NEXT:    or %s11, 0, %s9
  %r = mul i16 %b, %a
  ret i16 %r
}

define zeroext i32 @func32z(i32 zeroext %a, i32 zeroext %b) {
; CHECK-LABEL: func32z:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    muls.w.sx %s0, %s1, %s0
; CHECK-NEXT:    adds.w.zx %s0, %s0, (0)1
; CHECK-NEXT:    or %s11, 0, %s9
  %r = mul i32 %b, %a
  ret i32 %r
}

define i64 @func64z(i64 %a, i64 %b) {
; CHECK-LABEL: func64z:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    muls.l %s0, %s1, %s0
; CHECK-NEXT:    or %s11, 0, %s9
  %r = mul i64 %b, %a
  ret i64 %r
}

define i128 @func128z(i128 %a, i128 %b) {
; CHECK-LABEL: func128z:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    or %s4, 0, %s1
; CHECK-NEXT:    or %s5, 0, %s0
; CHECK-NEXT:    lea %s0, __multi3@lo
; CHECK-NEXT:    and %s0, %s0, (32)0
; CHECK-NEXT:    lea.sl %s12, __multi3@hi(, %s0)
; CHECK-NEXT:    or %s0, 0, %s2
; CHECK-NEXT:    or %s1, 0, %s3
; CHECK-NEXT:    or %s2, 0, %s5
; CHECK-NEXT:    or %s3, 0, %s4
; CHECK-NEXT:    bsic %s10, (, %s12)
; CHECK-NEXT:    or %s11, 0, %s9
  %r = mul i128 %b, %a
  ret i128 %r
}

define signext i8 @funci8s(i8 signext %a) {
; CHECK-LABEL: funci8s:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    muls.w.sx %s0, 5, %s0
; CHECK-NEXT:    sll %s0, %s0, 56
; CHECK-NEXT:    sra.l %s0, %s0, 56
; CHECK-NEXT:    or %s11, 0, %s9
  %r = mul i8 %a, 5
  ret i8 %r
}

define signext i16 @funci16s(i16 signext %a) {
; CHECK-LABEL: funci16s:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    muls.w.sx %s0, 5, %s0
; CHECK-NEXT:    sll %s0, %s0, 48
; CHECK-NEXT:    sra.l %s0, %s0, 48
; CHECK-NEXT:    or %s11, 0, %s9
  %r = mul i16 %a, 5
  ret i16 %r
}

define signext i32 @funci32s(i32 signext %a) {
; CHECK-LABEL: funci32s:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    muls.w.sx %s0, 5, %s0
; CHECK-NEXT:    adds.w.sx %s0, %s0, (0)1
; CHECK-NEXT:    or %s11, 0, %s9
  %r = mul nsw i32 %a, 5
  ret i32 %r
}

define i64 @funci64(i64 %a) {
; CHECK-LABEL: funci64:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    muls.l %s0, 5, %s0
; CHECK-NEXT:    or %s11, 0, %s9
  %r = mul nsw i64 %a, 5
  ret i64 %r
}

define i128 @funci128(i128 %a) {
; CHECK-LABEL: funci128:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    lea %s2, __multi3@lo
; CHECK-NEXT:    and %s2, %s2, (32)0
; CHECK-NEXT:    lea.sl %s12, __multi3@hi(, %s2)
; CHECK-NEXT:    or %s2, 5, (0)1
; CHECK-NEXT:    or %s3, 0, (0)1
; CHECK-NEXT:    bsic %s10, (, %s12)
; CHECK-NEXT:    or %s11, 0, %s9
  %r = mul nsw i128 %a, 5
  ret i128 %r
}

define zeroext i8 @funci8z(i8 zeroext %a) {
; CHECK-LABEL: funci8z:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    muls.w.sx %s0, 5, %s0
; CHECK-NEXT:    and %s0, %s0, (56)0
; CHECK-NEXT:    or %s11, 0, %s9
  %r = mul i8 %a, 5
  ret i8 %r
}

define zeroext i16 @funci16z(i16 zeroext %a) {
; CHECK-LABEL: funci16z:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    muls.w.sx %s0, 5, %s0
; CHECK-NEXT:    and %s0, %s0, (48)0
; CHECK-NEXT:    or %s11, 0, %s9
  %r = mul i16 %a, 5
  ret i16 %r
}

define zeroext i32 @funci32z(i32 zeroext %a) {
; CHECK-LABEL: funci32z:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    muls.w.sx %s0, 5, %s0
; CHECK-NEXT:    adds.w.zx %s0, %s0, (0)1
; CHECK-NEXT:    or %s11, 0, %s9
  %r = mul i32 %a, 5
  ret i32 %r
}

define i64 @funci64z(i64 %a) {
; CHECK-LABEL: funci64z:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    muls.l %s0, 5, %s0
; CHECK-NEXT:    or %s11, 0, %s9
  %r = mul i64 %a, 5
  ret i64 %r
}

define i128 @funci128z(i128 %a) {
; CHECK-LABEL: funci128z:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    lea %s2, __multi3@lo
; CHECK-NEXT:    and %s2, %s2, (32)0
; CHECK-NEXT:    lea.sl %s12, __multi3@hi(, %s2)
; CHECK-NEXT:    or %s2, 5, (0)1
; CHECK-NEXT:    or %s3, 0, (0)1
; CHECK-NEXT:    bsic %s10, (, %s12)
; CHECK-NEXT:    or %s11, 0, %s9
  %r = mul i128 %a, 5
  ret i128 %r
}

define zeroext i32 @funci32z_2(i32 zeroext %a) {
; CHECK-LABEL: funci32z_2:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    sla.w.sx %s0, %s0, 31
; CHECK-NEXT:    adds.w.zx %s0, %s0, (0)1
; CHECK-NEXT:    or %s11, 0, %s9
  %r = shl i32 %a, 31
  ret i32 %r
}

define i64 @funci64_2(i64 %a) {
; CHECK-LABEL: funci64_2:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    sll %s0, %s0, 31
; CHECK-NEXT:    or %s11, 0, %s9
  %r = shl nsw i64 %a, 31
  ret i64 %r
}

define i128 @funci128_2(i128 %a) {
; CHECK-LABEL: funci128_2:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    srl %s2, %s0, 33
; CHECK-NEXT:    sll %s1, %s1, 31
; CHECK-NEXT:    or %s1, %s1, %s2
; CHECK-NEXT:    sll %s0, %s0, 31
; CHECK-NEXT:    or %s11, 0, %s9
  %r = shl nsw i128 %a, 31
  ret i128 %r
}
