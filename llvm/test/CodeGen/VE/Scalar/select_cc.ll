; RUN: llc < %s -mtriple=ve | FileCheck %s

;;; Test all combination of input type and output type among following types.
;;;
;;; Types:
;;;  i1/i8/u8/i16/u16/i32/u32/i64/u64/i128/u128/float/double/fp128

; Function Attrs: norecurse nounwind readnone
define zeroext i1 @select_cc_i1_i1(i1 zeroext %0, i1 zeroext %1, i1 zeroext %2, i1 zeroext %3) {
; CHECK-LABEL: select_cc_i1_i1:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    xor %s0, %s0, %s1
; CHECK-NEXT:    cmov.w.ne %s2, %s3, %s0
; CHECK-NEXT:    adds.w.zx %s0, %s2, (0)1
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = xor i1 %0, %1
  %6 = select i1 %5, i1 %3, i1 %2
  ret i1 %6
}

; Function Attrs: norecurse nounwind readnone
define zeroext i1 @select_cc_i8_i1(i8 signext %0, i8 signext %1, i1 zeroext %2, i1 zeroext %3) {
; CHECK-LABEL: select_cc_i8_i1:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    cmps.w.sx %s0, %s0, %s1
; CHECK-NEXT:    cmov.w.eq %s3, %s2, %s0
; CHECK-NEXT:    adds.w.zx %s0, %s3, (0)1
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = icmp eq i8 %0, %1
  %6 = select i1 %5, i1 %2, i1 %3
  ret i1 %6
}

; Function Attrs: norecurse nounwind readnone
define zeroext i1 @select_cc_u8_i1(i8 zeroext %0, i8 zeroext %1, i1 zeroext %2, i1 zeroext %3) {
; CHECK-LABEL: select_cc_u8_i1:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    cmps.w.sx %s0, %s0, %s1
; CHECK-NEXT:    cmov.w.eq %s3, %s2, %s0
; CHECK-NEXT:    adds.w.zx %s0, %s3, (0)1
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = icmp eq i8 %0, %1
  %6 = select i1 %5, i1 %2, i1 %3
  ret i1 %6
}

; Function Attrs: norecurse nounwind readnone
define zeroext i1 @select_cc_i16_i1(i16 signext %0, i16 signext %1, i1 zeroext %2, i1 zeroext %3) {
; CHECK-LABEL: select_cc_i16_i1:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    cmps.w.sx %s0, %s0, %s1
; CHECK-NEXT:    cmov.w.eq %s3, %s2, %s0
; CHECK-NEXT:    adds.w.zx %s0, %s3, (0)1
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = icmp eq i16 %0, %1
  %6 = select i1 %5, i1 %2, i1 %3
  ret i1 %6
}

; Function Attrs: norecurse nounwind readnone
define zeroext i1 @select_cc_u16_i1(i16 zeroext %0, i16 zeroext %1, i1 zeroext %2, i1 zeroext %3) {
; CHECK-LABEL: select_cc_u16_i1:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    cmps.w.sx %s0, %s0, %s1
; CHECK-NEXT:    cmov.w.eq %s3, %s2, %s0
; CHECK-NEXT:    adds.w.zx %s0, %s3, (0)1
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = icmp eq i16 %0, %1
  %6 = select i1 %5, i1 %2, i1 %3
  ret i1 %6
}

; Function Attrs: norecurse nounwind readnone
define zeroext i1 @select_cc_i32_i1(i32 signext %0, i32 signext %1, i1 zeroext %2, i1 zeroext %3) {
; CHECK-LABEL: select_cc_i32_i1:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    cmps.w.sx %s0, %s0, %s1
; CHECK-NEXT:    cmov.w.eq %s3, %s2, %s0
; CHECK-NEXT:    adds.w.zx %s0, %s3, (0)1
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = icmp eq i32 %0, %1
  %6 = select i1 %5, i1 %2, i1 %3
  ret i1 %6
}

; Function Attrs: norecurse nounwind readnone
define zeroext i1 @select_cc_u32_i1(i32 zeroext %0, i32 zeroext %1, i1 zeroext %2, i1 zeroext %3) {
; CHECK-LABEL: select_cc_u32_i1:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    cmps.w.sx %s0, %s0, %s1
; CHECK-NEXT:    cmov.w.eq %s3, %s2, %s0
; CHECK-NEXT:    adds.w.zx %s0, %s3, (0)1
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = icmp eq i32 %0, %1
  %6 = select i1 %5, i1 %2, i1 %3
  ret i1 %6
}

; Function Attrs: norecurse nounwind readnone
define zeroext i1 @select_cc_i64_i1(i64 %0, i64 %1, i1 zeroext %2, i1 zeroext %3) {
; CHECK-LABEL: select_cc_i64_i1:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    cmps.l %s0, %s0, %s1
; CHECK-NEXT:    cmov.l.eq %s3, %s2, %s0
; CHECK-NEXT:    adds.w.zx %s0, %s3, (0)1
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = icmp eq i64 %0, %1
  %6 = select i1 %5, i1 %2, i1 %3
  ret i1 %6
}

; Function Attrs: norecurse nounwind readnone
define zeroext i1 @select_cc_u64_i1(i64 %0, i64 %1, i1 zeroext %2, i1 zeroext %3) {
; CHECK-LABEL: select_cc_u64_i1:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    cmps.l %s0, %s0, %s1
; CHECK-NEXT:    cmov.l.eq %s3, %s2, %s0
; CHECK-NEXT:    adds.w.zx %s0, %s3, (0)1
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = icmp eq i64 %0, %1
  %6 = select i1 %5, i1 %2, i1 %3
  ret i1 %6
}

; Function Attrs: norecurse nounwind readnone
define zeroext i1 @select_cc_i128_i1(i128 %0, i128 %1, i1 zeroext %2, i1 zeroext %3) {
; CHECK-LABEL: select_cc_i128_i1:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    xor %s1, %s1, %s3
; CHECK-NEXT:    xor %s0, %s0, %s2
; CHECK-NEXT:    or %s0, %s0, %s1
; CHECK-NEXT:    cmps.l %s0, %s0, (0)1
; CHECK-NEXT:    cmov.l.eq %s5, %s4, %s0
; CHECK-NEXT:    adds.w.zx %s0, %s5, (0)1
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = icmp eq i128 %0, %1
  %6 = select i1 %5, i1 %2, i1 %3
  ret i1 %6
}

; Function Attrs: norecurse nounwind readnone
define zeroext i1 @select_cc_u128_i1(i128 %0, i128 %1, i1 zeroext %2, i1 zeroext %3) {
; CHECK-LABEL: select_cc_u128_i1:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    xor %s1, %s1, %s3
; CHECK-NEXT:    xor %s0, %s0, %s2
; CHECK-NEXT:    or %s0, %s0, %s1
; CHECK-NEXT:    cmps.l %s0, %s0, (0)1
; CHECK-NEXT:    cmov.l.eq %s5, %s4, %s0
; CHECK-NEXT:    adds.w.zx %s0, %s5, (0)1
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = icmp eq i128 %0, %1
  %6 = select i1 %5, i1 %2, i1 %3
  ret i1 %6
}

; Function Attrs: norecurse nounwind readnone
define zeroext i1 @select_cc_float_i1(float %0, float %1, i1 zeroext %2, i1 zeroext %3) {
; CHECK-LABEL: select_cc_float_i1:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    fcmp.s %s0, %s0, %s1
; CHECK-NEXT:    cmov.s.eq %s3, %s2, %s0
; CHECK-NEXT:    adds.w.zx %s0, %s3, (0)1
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = fcmp fast oeq float %0, %1
  %6 = select i1 %5, i1 %2, i1 %3
  ret i1 %6
}

; Function Attrs: norecurse nounwind readnone
define zeroext i1 @select_cc_double_i1(double %0, double %1, i1 zeroext %2, i1 zeroext %3) {
; CHECK-LABEL: select_cc_double_i1:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    fcmp.d %s0, %s0, %s1
; CHECK-NEXT:    cmov.d.eq %s3, %s2, %s0
; CHECK-NEXT:    adds.w.zx %s0, %s3, (0)1
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = fcmp fast oeq double %0, %1
  %6 = select i1 %5, i1 %2, i1 %3
  ret i1 %6
}

; Function Attrs: norecurse nounwind readnone
define zeroext i1 @select_cc_quad_i1(fp128 %0, fp128 %1, i1 zeroext %2, i1 zeroext %3) {
; CHECK-LABEL: select_cc_quad_i1:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    fcmp.q %s0, %s0, %s2
; CHECK-NEXT:    cmov.d.eq %s5, %s4, %s0
; CHECK-NEXT:    adds.w.zx %s0, %s5, (0)1
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = fcmp fast oeq fp128 %0, %1
  %6 = select i1 %5, i1 %2, i1 %3
  ret i1 %6
}

; Function Attrs: norecurse nounwind readnone
define signext i8 @select_cc_i1_i8(i1 zeroext %0, i1 zeroext %1, i8 signext %2, i8 signext %3) {
; CHECK-LABEL: select_cc_i1_i8:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    xor %s0, %s0, %s1
; CHECK-NEXT:    cmov.w.ne %s2, %s3, %s0
; CHECK-NEXT:    adds.w.sx %s0, %s2, (0)1
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = xor i1 %0, %1
  %6 = select i1 %5, i8 %3, i8 %2
  ret i8 %6
}

; Function Attrs: norecurse nounwind readnone
define signext i8 @select_cc_i8_i8(i8 signext %0, i8 signext %1, i8 signext %2, i8 signext %3) {
; CHECK-LABEL: select_cc_i8_i8:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    cmps.w.sx %s0, %s0, %s1
; CHECK-NEXT:    cmov.w.eq %s3, %s2, %s0
; CHECK-NEXT:    adds.w.sx %s0, %s3, (0)1
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = icmp eq i8 %0, %1
  %6 = select i1 %5, i8 %2, i8 %3
  ret i8 %6
}

; Function Attrs: norecurse nounwind readnone
define signext i8 @select_cc_u8_i8(i8 zeroext %0, i8 zeroext %1, i8 signext %2, i8 signext %3) {
; CHECK-LABEL: select_cc_u8_i8:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    cmps.w.sx %s0, %s0, %s1
; CHECK-NEXT:    cmov.w.eq %s3, %s2, %s0
; CHECK-NEXT:    adds.w.sx %s0, %s3, (0)1
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = icmp eq i8 %0, %1
  %6 = select i1 %5, i8 %2, i8 %3
  ret i8 %6
}

; Function Attrs: norecurse nounwind readnone
define signext i8 @select_cc_i16_i8(i16 signext %0, i16 signext %1, i8 signext %2, i8 signext %3) {
; CHECK-LABEL: select_cc_i16_i8:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    cmps.w.sx %s0, %s0, %s1
; CHECK-NEXT:    cmov.w.eq %s3, %s2, %s0
; CHECK-NEXT:    adds.w.sx %s0, %s3, (0)1
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = icmp eq i16 %0, %1
  %6 = select i1 %5, i8 %2, i8 %3
  ret i8 %6
}

; Function Attrs: norecurse nounwind readnone
define signext i8 @select_cc_u16_i8(i16 zeroext %0, i16 zeroext %1, i8 signext %2, i8 signext %3) {
; CHECK-LABEL: select_cc_u16_i8:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    cmps.w.sx %s0, %s0, %s1
; CHECK-NEXT:    cmov.w.eq %s3, %s2, %s0
; CHECK-NEXT:    adds.w.sx %s0, %s3, (0)1
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = icmp eq i16 %0, %1
  %6 = select i1 %5, i8 %2, i8 %3
  ret i8 %6
}

; Function Attrs: norecurse nounwind readnone
define signext i8 @select_cc_i32_i8(i32 signext %0, i32 signext %1, i8 signext %2, i8 signext %3) {
; CHECK-LABEL: select_cc_i32_i8:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    cmps.w.sx %s0, %s0, %s1
; CHECK-NEXT:    cmov.w.eq %s3, %s2, %s0
; CHECK-NEXT:    adds.w.sx %s0, %s3, (0)1
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = icmp eq i32 %0, %1
  %6 = select i1 %5, i8 %2, i8 %3
  ret i8 %6
}

; Function Attrs: norecurse nounwind readnone
define signext i8 @select_cc_u32_i8(i32 zeroext %0, i32 zeroext %1, i8 signext %2, i8 signext %3) {
; CHECK-LABEL: select_cc_u32_i8:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    cmps.w.sx %s0, %s0, %s1
; CHECK-NEXT:    cmov.w.eq %s3, %s2, %s0
; CHECK-NEXT:    adds.w.sx %s0, %s3, (0)1
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = icmp eq i32 %0, %1
  %6 = select i1 %5, i8 %2, i8 %3
  ret i8 %6
}

; Function Attrs: norecurse nounwind readnone
define signext i8 @select_cc_i64_i8(i64 %0, i64 %1, i8 signext %2, i8 signext %3) {
; CHECK-LABEL: select_cc_i64_i8:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    cmps.l %s0, %s0, %s1
; CHECK-NEXT:    cmov.l.eq %s3, %s2, %s0
; CHECK-NEXT:    adds.w.sx %s0, %s3, (0)1
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = icmp eq i64 %0, %1
  %6 = select i1 %5, i8 %2, i8 %3
  ret i8 %6
}

; Function Attrs: norecurse nounwind readnone
define signext i8 @select_cc_u64_i8(i64 %0, i64 %1, i8 signext %2, i8 signext %3) {
; CHECK-LABEL: select_cc_u64_i8:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    cmps.l %s0, %s0, %s1
; CHECK-NEXT:    cmov.l.eq %s3, %s2, %s0
; CHECK-NEXT:    adds.w.sx %s0, %s3, (0)1
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = icmp eq i64 %0, %1
  %6 = select i1 %5, i8 %2, i8 %3
  ret i8 %6
}

; Function Attrs: norecurse nounwind readnone
define signext i8 @select_cc_i128_i8(i128 %0, i128 %1, i8 signext %2, i8 signext %3) {
; CHECK-LABEL: select_cc_i128_i8:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    xor %s1, %s1, %s3
; CHECK-NEXT:    xor %s0, %s0, %s2
; CHECK-NEXT:    or %s0, %s0, %s1
; CHECK-NEXT:    cmps.l %s0, %s0, (0)1
; CHECK-NEXT:    cmov.l.eq %s5, %s4, %s0
; CHECK-NEXT:    adds.w.sx %s0, %s5, (0)1
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = icmp eq i128 %0, %1
  %6 = select i1 %5, i8 %2, i8 %3
  ret i8 %6
}

; Function Attrs: norecurse nounwind readnone
define signext i8 @select_cc_u128_i8(i128 %0, i128 %1, i8 signext %2, i8 signext %3) {
; CHECK-LABEL: select_cc_u128_i8:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    xor %s1, %s1, %s3
; CHECK-NEXT:    xor %s0, %s0, %s2
; CHECK-NEXT:    or %s0, %s0, %s1
; CHECK-NEXT:    cmps.l %s0, %s0, (0)1
; CHECK-NEXT:    cmov.l.eq %s5, %s4, %s0
; CHECK-NEXT:    adds.w.sx %s0, %s5, (0)1
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = icmp eq i128 %0, %1
  %6 = select i1 %5, i8 %2, i8 %3
  ret i8 %6
}

; Function Attrs: norecurse nounwind readnone
define signext i8 @select_cc_float_i8(float %0, float %1, i8 signext %2, i8 signext %3) {
; CHECK-LABEL: select_cc_float_i8:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    fcmp.s %s0, %s0, %s1
; CHECK-NEXT:    cmov.s.eq %s3, %s2, %s0
; CHECK-NEXT:    adds.w.sx %s0, %s3, (0)1
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = fcmp fast oeq float %0, %1
  %6 = select i1 %5, i8 %2, i8 %3
  ret i8 %6
}

; Function Attrs: norecurse nounwind readnone
define signext i8 @select_cc_double_i8(double %0, double %1, i8 signext %2, i8 signext %3) {
; CHECK-LABEL: select_cc_double_i8:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    fcmp.d %s0, %s0, %s1
; CHECK-NEXT:    cmov.d.eq %s3, %s2, %s0
; CHECK-NEXT:    adds.w.sx %s0, %s3, (0)1
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = fcmp fast oeq double %0, %1
  %6 = select i1 %5, i8 %2, i8 %3
  ret i8 %6
}

; Function Attrs: norecurse nounwind readnone
define signext i8 @select_cc_quad_i8(fp128 %0, fp128 %1, i8 signext %2, i8 signext %3) {
; CHECK-LABEL: select_cc_quad_i8:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    fcmp.q %s0, %s0, %s2
; CHECK-NEXT:    cmov.d.eq %s5, %s4, %s0
; CHECK-NEXT:    adds.w.sx %s0, %s5, (0)1
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = fcmp fast oeq fp128 %0, %1
  %6 = select i1 %5, i8 %2, i8 %3
  ret i8 %6
}

; Function Attrs: norecurse nounwind readnone
define zeroext i8 @select_cc_i1_u8(i1 zeroext %0, i1 zeroext %1, i8 zeroext %2, i8 zeroext %3) {
; CHECK-LABEL: select_cc_i1_u8:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    xor %s0, %s0, %s1
; CHECK-NEXT:    cmov.w.ne %s2, %s3, %s0
; CHECK-NEXT:    adds.w.zx %s0, %s2, (0)1
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = xor i1 %0, %1
  %6 = select i1 %5, i8 %3, i8 %2
  ret i8 %6
}

; Function Attrs: norecurse nounwind readnone
define zeroext i8 @select_cc_i8_u8(i8 signext %0, i8 signext %1, i8 zeroext %2, i8 zeroext %3) {
; CHECK-LABEL: select_cc_i8_u8:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    cmps.w.sx %s0, %s0, %s1
; CHECK-NEXT:    cmov.w.eq %s3, %s2, %s0
; CHECK-NEXT:    adds.w.zx %s0, %s3, (0)1
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = icmp eq i8 %0, %1
  %6 = select i1 %5, i8 %2, i8 %3
  ret i8 %6
}

; Function Attrs: norecurse nounwind readnone
define zeroext i8 @select_cc_u8_u8(i8 zeroext %0, i8 zeroext %1, i8 zeroext %2, i8 zeroext %3) {
; CHECK-LABEL: select_cc_u8_u8:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    cmps.w.sx %s0, %s0, %s1
; CHECK-NEXT:    cmov.w.eq %s3, %s2, %s0
; CHECK-NEXT:    adds.w.zx %s0, %s3, (0)1
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = icmp eq i8 %0, %1
  %6 = select i1 %5, i8 %2, i8 %3
  ret i8 %6
}

; Function Attrs: norecurse nounwind readnone
define zeroext i8 @select_cc_i16_u8(i16 signext %0, i16 signext %1, i8 zeroext %2, i8 zeroext %3) {
; CHECK-LABEL: select_cc_i16_u8:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    cmps.w.sx %s0, %s0, %s1
; CHECK-NEXT:    cmov.w.eq %s3, %s2, %s0
; CHECK-NEXT:    adds.w.zx %s0, %s3, (0)1
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = icmp eq i16 %0, %1
  %6 = select i1 %5, i8 %2, i8 %3
  ret i8 %6
}

; Function Attrs: norecurse nounwind readnone
define zeroext i8 @select_cc_u16_u8(i16 zeroext %0, i16 zeroext %1, i8 zeroext %2, i8 zeroext %3) {
; CHECK-LABEL: select_cc_u16_u8:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    cmps.w.sx %s0, %s0, %s1
; CHECK-NEXT:    cmov.w.eq %s3, %s2, %s0
; CHECK-NEXT:    adds.w.zx %s0, %s3, (0)1
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = icmp eq i16 %0, %1
  %6 = select i1 %5, i8 %2, i8 %3
  ret i8 %6
}

; Function Attrs: norecurse nounwind readnone
define zeroext i8 @select_cc_i32_u8(i32 signext %0, i32 signext %1, i8 zeroext %2, i8 zeroext %3) {
; CHECK-LABEL: select_cc_i32_u8:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    cmps.w.sx %s0, %s0, %s1
; CHECK-NEXT:    cmov.w.eq %s3, %s2, %s0
; CHECK-NEXT:    adds.w.zx %s0, %s3, (0)1
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = icmp eq i32 %0, %1
  %6 = select i1 %5, i8 %2, i8 %3
  ret i8 %6
}

; Function Attrs: norecurse nounwind readnone
define zeroext i8 @select_cc_u32_u8(i32 zeroext %0, i32 zeroext %1, i8 zeroext %2, i8 zeroext %3) {
; CHECK-LABEL: select_cc_u32_u8:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    cmps.w.sx %s0, %s0, %s1
; CHECK-NEXT:    cmov.w.eq %s3, %s2, %s0
; CHECK-NEXT:    adds.w.zx %s0, %s3, (0)1
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = icmp eq i32 %0, %1
  %6 = select i1 %5, i8 %2, i8 %3
  ret i8 %6
}

; Function Attrs: norecurse nounwind readnone
define zeroext i8 @select_cc_i64_u8(i64 %0, i64 %1, i8 zeroext %2, i8 zeroext %3) {
; CHECK-LABEL: select_cc_i64_u8:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    cmps.l %s0, %s0, %s1
; CHECK-NEXT:    cmov.l.eq %s3, %s2, %s0
; CHECK-NEXT:    adds.w.zx %s0, %s3, (0)1
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = icmp eq i64 %0, %1
  %6 = select i1 %5, i8 %2, i8 %3
  ret i8 %6
}

; Function Attrs: norecurse nounwind readnone
define zeroext i8 @select_cc_u64_u8(i64 %0, i64 %1, i8 zeroext %2, i8 zeroext %3) {
; CHECK-LABEL: select_cc_u64_u8:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    cmps.l %s0, %s0, %s1
; CHECK-NEXT:    cmov.l.eq %s3, %s2, %s0
; CHECK-NEXT:    adds.w.zx %s0, %s3, (0)1
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = icmp eq i64 %0, %1
  %6 = select i1 %5, i8 %2, i8 %3
  ret i8 %6
}

; Function Attrs: norecurse nounwind readnone
define zeroext i8 @select_cc_i128_u8(i128 %0, i128 %1, i8 zeroext %2, i8 zeroext %3) {
; CHECK-LABEL: select_cc_i128_u8:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    xor %s1, %s1, %s3
; CHECK-NEXT:    xor %s0, %s0, %s2
; CHECK-NEXT:    or %s0, %s0, %s1
; CHECK-NEXT:    cmps.l %s0, %s0, (0)1
; CHECK-NEXT:    cmov.l.eq %s5, %s4, %s0
; CHECK-NEXT:    adds.w.zx %s0, %s5, (0)1
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = icmp eq i128 %0, %1
  %6 = select i1 %5, i8 %2, i8 %3
  ret i8 %6
}

; Function Attrs: norecurse nounwind readnone
define zeroext i8 @select_cc_u128_u8(i128 %0, i128 %1, i8 zeroext %2, i8 zeroext %3) {
; CHECK-LABEL: select_cc_u128_u8:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    xor %s1, %s1, %s3
; CHECK-NEXT:    xor %s0, %s0, %s2
; CHECK-NEXT:    or %s0, %s0, %s1
; CHECK-NEXT:    cmps.l %s0, %s0, (0)1
; CHECK-NEXT:    cmov.l.eq %s5, %s4, %s0
; CHECK-NEXT:    adds.w.zx %s0, %s5, (0)1
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = icmp eq i128 %0, %1
  %6 = select i1 %5, i8 %2, i8 %3
  ret i8 %6
}

; Function Attrs: norecurse nounwind readnone
define zeroext i8 @select_cc_float_u8(float %0, float %1, i8 zeroext %2, i8 zeroext %3) {
; CHECK-LABEL: select_cc_float_u8:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    fcmp.s %s0, %s0, %s1
; CHECK-NEXT:    cmov.s.eq %s3, %s2, %s0
; CHECK-NEXT:    adds.w.zx %s0, %s3, (0)1
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = fcmp fast oeq float %0, %1
  %6 = select i1 %5, i8 %2, i8 %3
  ret i8 %6
}

; Function Attrs: norecurse nounwind readnone
define zeroext i8 @select_cc_double_u8(double %0, double %1, i8 zeroext %2, i8 zeroext %3) {
; CHECK-LABEL: select_cc_double_u8:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    fcmp.d %s0, %s0, %s1
; CHECK-NEXT:    cmov.d.eq %s3, %s2, %s0
; CHECK-NEXT:    adds.w.zx %s0, %s3, (0)1
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = fcmp fast oeq double %0, %1
  %6 = select i1 %5, i8 %2, i8 %3
  ret i8 %6
}

; Function Attrs: norecurse nounwind readnone
define zeroext i8 @select_cc_quad_u8(fp128 %0, fp128 %1, i8 zeroext %2, i8 zeroext %3) {
; CHECK-LABEL: select_cc_quad_u8:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    fcmp.q %s0, %s0, %s2
; CHECK-NEXT:    cmov.d.eq %s5, %s4, %s0
; CHECK-NEXT:    adds.w.zx %s0, %s5, (0)1
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = fcmp fast oeq fp128 %0, %1
  %6 = select i1 %5, i8 %2, i8 %3
  ret i8 %6
}

; Function Attrs: norecurse nounwind readnone
define signext i16 @select_cc_i1_i16(i1 zeroext %0, i1 zeroext %1, i16 signext %2, i16 signext %3) {
; CHECK-LABEL: select_cc_i1_i16:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    xor %s0, %s0, %s1
; CHECK-NEXT:    cmov.w.ne %s2, %s3, %s0
; CHECK-NEXT:    adds.w.sx %s0, %s2, (0)1
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = xor i1 %0, %1
  %6 = select i1 %5, i16 %3, i16 %2
  ret i16 %6
}

; Function Attrs: norecurse nounwind readnone
define signext i16 @select_cc_i8_i16(i8 signext %0, i8 signext %1, i16 signext %2, i16 signext %3) {
; CHECK-LABEL: select_cc_i8_i16:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    cmps.w.sx %s0, %s0, %s1
; CHECK-NEXT:    cmov.w.eq %s3, %s2, %s0
; CHECK-NEXT:    adds.w.sx %s0, %s3, (0)1
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = icmp eq i8 %0, %1
  %6 = select i1 %5, i16 %2, i16 %3
  ret i16 %6
}

; Function Attrs: norecurse nounwind readnone
define signext i16 @select_cc_u8_i16(i8 zeroext %0, i8 zeroext %1, i16 signext %2, i16 signext %3) {
; CHECK-LABEL: select_cc_u8_i16:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    cmps.w.sx %s0, %s0, %s1
; CHECK-NEXT:    cmov.w.eq %s3, %s2, %s0
; CHECK-NEXT:    adds.w.sx %s0, %s3, (0)1
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = icmp eq i8 %0, %1
  %6 = select i1 %5, i16 %2, i16 %3
  ret i16 %6
}

; Function Attrs: norecurse nounwind readnone
define signext i16 @select_cc_i16_i16(i16 signext %0, i16 signext %1, i16 signext %2, i16 signext %3) {
; CHECK-LABEL: select_cc_i16_i16:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    cmps.w.sx %s0, %s0, %s1
; CHECK-NEXT:    cmov.w.eq %s3, %s2, %s0
; CHECK-NEXT:    adds.w.sx %s0, %s3, (0)1
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = icmp eq i16 %0, %1
  %6 = select i1 %5, i16 %2, i16 %3
  ret i16 %6
}

; Function Attrs: norecurse nounwind readnone
define signext i16 @select_cc_u16_i16(i16 zeroext %0, i16 zeroext %1, i16 signext %2, i16 signext %3) {
; CHECK-LABEL: select_cc_u16_i16:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    cmps.w.sx %s0, %s0, %s1
; CHECK-NEXT:    cmov.w.eq %s3, %s2, %s0
; CHECK-NEXT:    adds.w.sx %s0, %s3, (0)1
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = icmp eq i16 %0, %1
  %6 = select i1 %5, i16 %2, i16 %3
  ret i16 %6
}

; Function Attrs: norecurse nounwind readnone
define signext i16 @select_cc_i32_i16(i32 signext %0, i32 signext %1, i16 signext %2, i16 signext %3) {
; CHECK-LABEL: select_cc_i32_i16:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    cmps.w.sx %s0, %s0, %s1
; CHECK-NEXT:    cmov.w.eq %s3, %s2, %s0
; CHECK-NEXT:    adds.w.sx %s0, %s3, (0)1
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = icmp eq i32 %0, %1
  %6 = select i1 %5, i16 %2, i16 %3
  ret i16 %6
}

; Function Attrs: norecurse nounwind readnone
define signext i16 @select_cc_u32_i16(i32 zeroext %0, i32 zeroext %1, i16 signext %2, i16 signext %3) {
; CHECK-LABEL: select_cc_u32_i16:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    cmps.w.sx %s0, %s0, %s1
; CHECK-NEXT:    cmov.w.eq %s3, %s2, %s0
; CHECK-NEXT:    adds.w.sx %s0, %s3, (0)1
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = icmp eq i32 %0, %1
  %6 = select i1 %5, i16 %2, i16 %3
  ret i16 %6
}

; Function Attrs: norecurse nounwind readnone
define signext i16 @select_cc_i64_i16(i64 %0, i64 %1, i16 signext %2, i16 signext %3) {
; CHECK-LABEL: select_cc_i64_i16:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    cmps.l %s0, %s0, %s1
; CHECK-NEXT:    cmov.l.eq %s3, %s2, %s0
; CHECK-NEXT:    adds.w.sx %s0, %s3, (0)1
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = icmp eq i64 %0, %1
  %6 = select i1 %5, i16 %2, i16 %3
  ret i16 %6
}

; Function Attrs: norecurse nounwind readnone
define signext i16 @select_cc_u64_i16(i64 %0, i64 %1, i16 signext %2, i16 signext %3) {
; CHECK-LABEL: select_cc_u64_i16:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    cmps.l %s0, %s0, %s1
; CHECK-NEXT:    cmov.l.eq %s3, %s2, %s0
; CHECK-NEXT:    adds.w.sx %s0, %s3, (0)1
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = icmp eq i64 %0, %1
  %6 = select i1 %5, i16 %2, i16 %3
  ret i16 %6
}

; Function Attrs: norecurse nounwind readnone
define signext i16 @select_cc_i128_i16(i128 %0, i128 %1, i16 signext %2, i16 signext %3) {
; CHECK-LABEL: select_cc_i128_i16:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    xor %s1, %s1, %s3
; CHECK-NEXT:    xor %s0, %s0, %s2
; CHECK-NEXT:    or %s0, %s0, %s1
; CHECK-NEXT:    cmps.l %s0, %s0, (0)1
; CHECK-NEXT:    cmov.l.eq %s5, %s4, %s0
; CHECK-NEXT:    adds.w.sx %s0, %s5, (0)1
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = icmp eq i128 %0, %1
  %6 = select i1 %5, i16 %2, i16 %3
  ret i16 %6
}

; Function Attrs: norecurse nounwind readnone
define signext i16 @select_cc_u128_i16(i128 %0, i128 %1, i16 signext %2, i16 signext %3) {
; CHECK-LABEL: select_cc_u128_i16:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    xor %s1, %s1, %s3
; CHECK-NEXT:    xor %s0, %s0, %s2
; CHECK-NEXT:    or %s0, %s0, %s1
; CHECK-NEXT:    cmps.l %s0, %s0, (0)1
; CHECK-NEXT:    cmov.l.eq %s5, %s4, %s0
; CHECK-NEXT:    adds.w.sx %s0, %s5, (0)1
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = icmp eq i128 %0, %1
  %6 = select i1 %5, i16 %2, i16 %3
  ret i16 %6
}

; Function Attrs: norecurse nounwind readnone
define signext i16 @select_cc_float_i16(float %0, float %1, i16 signext %2, i16 signext %3) {
; CHECK-LABEL: select_cc_float_i16:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    fcmp.s %s0, %s0, %s1
; CHECK-NEXT:    cmov.s.eq %s3, %s2, %s0
; CHECK-NEXT:    adds.w.sx %s0, %s3, (0)1
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = fcmp fast oeq float %0, %1
  %6 = select i1 %5, i16 %2, i16 %3
  ret i16 %6
}

; Function Attrs: norecurse nounwind readnone
define signext i16 @select_cc_double_i16(double %0, double %1, i16 signext %2, i16 signext %3) {
; CHECK-LABEL: select_cc_double_i16:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    fcmp.d %s0, %s0, %s1
; CHECK-NEXT:    cmov.d.eq %s3, %s2, %s0
; CHECK-NEXT:    adds.w.sx %s0, %s3, (0)1
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = fcmp fast oeq double %0, %1
  %6 = select i1 %5, i16 %2, i16 %3
  ret i16 %6
}

; Function Attrs: norecurse nounwind readnone
define signext i16 @select_cc_quad_i16(fp128 %0, fp128 %1, i16 signext %2, i16 signext %3) {
; CHECK-LABEL: select_cc_quad_i16:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    fcmp.q %s0, %s0, %s2
; CHECK-NEXT:    cmov.d.eq %s5, %s4, %s0
; CHECK-NEXT:    adds.w.sx %s0, %s5, (0)1
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = fcmp fast oeq fp128 %0, %1
  %6 = select i1 %5, i16 %2, i16 %3
  ret i16 %6
}

; Function Attrs: norecurse nounwind readnone
define zeroext i16 @select_cc_i1_u16(i1 zeroext %0, i1 zeroext %1, i16 zeroext %2, i16 zeroext %3) {
; CHECK-LABEL: select_cc_i1_u16:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    xor %s0, %s0, %s1
; CHECK-NEXT:    cmov.w.ne %s2, %s3, %s0
; CHECK-NEXT:    adds.w.zx %s0, %s2, (0)1
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = xor i1 %0, %1
  %6 = select i1 %5, i16 %3, i16 %2
  ret i16 %6
}

; Function Attrs: norecurse nounwind readnone
define zeroext i16 @select_cc_i8_u16(i8 signext %0, i8 signext %1, i16 zeroext %2, i16 zeroext %3) {
; CHECK-LABEL: select_cc_i8_u16:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    cmps.w.sx %s0, %s0, %s1
; CHECK-NEXT:    cmov.w.eq %s3, %s2, %s0
; CHECK-NEXT:    adds.w.zx %s0, %s3, (0)1
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = icmp eq i8 %0, %1
  %6 = select i1 %5, i16 %2, i16 %3
  ret i16 %6
}

; Function Attrs: norecurse nounwind readnone
define zeroext i16 @select_cc_u8_u16(i8 zeroext %0, i8 zeroext %1, i16 zeroext %2, i16 zeroext %3) {
; CHECK-LABEL: select_cc_u8_u16:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    cmps.w.sx %s0, %s0, %s1
; CHECK-NEXT:    cmov.w.eq %s3, %s2, %s0
; CHECK-NEXT:    adds.w.zx %s0, %s3, (0)1
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = icmp eq i8 %0, %1
  %6 = select i1 %5, i16 %2, i16 %3
  ret i16 %6
}

; Function Attrs: norecurse nounwind readnone
define zeroext i16 @select_cc_i16_u16(i16 signext %0, i16 signext %1, i16 zeroext %2, i16 zeroext %3) {
; CHECK-LABEL: select_cc_i16_u16:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    cmps.w.sx %s0, %s0, %s1
; CHECK-NEXT:    cmov.w.eq %s3, %s2, %s0
; CHECK-NEXT:    adds.w.zx %s0, %s3, (0)1
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = icmp eq i16 %0, %1
  %6 = select i1 %5, i16 %2, i16 %3
  ret i16 %6
}

; Function Attrs: norecurse nounwind readnone
define zeroext i16 @select_cc_u16_u16(i16 zeroext %0, i16 zeroext %1, i16 zeroext %2, i16 zeroext %3) {
; CHECK-LABEL: select_cc_u16_u16:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    cmps.w.sx %s0, %s0, %s1
; CHECK-NEXT:    cmov.w.eq %s3, %s2, %s0
; CHECK-NEXT:    adds.w.zx %s0, %s3, (0)1
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = icmp eq i16 %0, %1
  %6 = select i1 %5, i16 %2, i16 %3
  ret i16 %6
}

; Function Attrs: norecurse nounwind readnone
define zeroext i16 @select_cc_i32_u16(i32 signext %0, i32 signext %1, i16 zeroext %2, i16 zeroext %3) {
; CHECK-LABEL: select_cc_i32_u16:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    cmps.w.sx %s0, %s0, %s1
; CHECK-NEXT:    cmov.w.eq %s3, %s2, %s0
; CHECK-NEXT:    adds.w.zx %s0, %s3, (0)1
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = icmp eq i32 %0, %1
  %6 = select i1 %5, i16 %2, i16 %3
  ret i16 %6
}

; Function Attrs: norecurse nounwind readnone
define zeroext i16 @select_cc_u32_u16(i32 zeroext %0, i32 zeroext %1, i16 zeroext %2, i16 zeroext %3) {
; CHECK-LABEL: select_cc_u32_u16:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    cmps.w.sx %s0, %s0, %s1
; CHECK-NEXT:    cmov.w.eq %s3, %s2, %s0
; CHECK-NEXT:    adds.w.zx %s0, %s3, (0)1
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = icmp eq i32 %0, %1
  %6 = select i1 %5, i16 %2, i16 %3
  ret i16 %6
}

; Function Attrs: norecurse nounwind readnone
define zeroext i16 @select_cc_i64_u16(i64 %0, i64 %1, i16 zeroext %2, i16 zeroext %3) {
; CHECK-LABEL: select_cc_i64_u16:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    cmps.l %s0, %s0, %s1
; CHECK-NEXT:    cmov.l.eq %s3, %s2, %s0
; CHECK-NEXT:    adds.w.zx %s0, %s3, (0)1
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = icmp eq i64 %0, %1
  %6 = select i1 %5, i16 %2, i16 %3
  ret i16 %6
}

; Function Attrs: norecurse nounwind readnone
define zeroext i16 @select_cc_u64_u16(i64 %0, i64 %1, i16 zeroext %2, i16 zeroext %3) {
; CHECK-LABEL: select_cc_u64_u16:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    cmps.l %s0, %s0, %s1
; CHECK-NEXT:    cmov.l.eq %s3, %s2, %s0
; CHECK-NEXT:    adds.w.zx %s0, %s3, (0)1
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = icmp eq i64 %0, %1
  %6 = select i1 %5, i16 %2, i16 %3
  ret i16 %6
}

; Function Attrs: norecurse nounwind readnone
define zeroext i16 @select_cc_i128_u16(i128 %0, i128 %1, i16 zeroext %2, i16 zeroext %3) {
; CHECK-LABEL: select_cc_i128_u16:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    xor %s1, %s1, %s3
; CHECK-NEXT:    xor %s0, %s0, %s2
; CHECK-NEXT:    or %s0, %s0, %s1
; CHECK-NEXT:    cmps.l %s0, %s0, (0)1
; CHECK-NEXT:    cmov.l.eq %s5, %s4, %s0
; CHECK-NEXT:    adds.w.zx %s0, %s5, (0)1
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = icmp eq i128 %0, %1
  %6 = select i1 %5, i16 %2, i16 %3
  ret i16 %6
}

; Function Attrs: norecurse nounwind readnone
define zeroext i16 @select_cc_u128_u16(i128 %0, i128 %1, i16 zeroext %2, i16 zeroext %3) {
; CHECK-LABEL: select_cc_u128_u16:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    xor %s1, %s1, %s3
; CHECK-NEXT:    xor %s0, %s0, %s2
; CHECK-NEXT:    or %s0, %s0, %s1
; CHECK-NEXT:    cmps.l %s0, %s0, (0)1
; CHECK-NEXT:    cmov.l.eq %s5, %s4, %s0
; CHECK-NEXT:    adds.w.zx %s0, %s5, (0)1
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = icmp eq i128 %0, %1
  %6 = select i1 %5, i16 %2, i16 %3
  ret i16 %6
}

; Function Attrs: norecurse nounwind readnone
define zeroext i16 @select_cc_float_u16(float %0, float %1, i16 zeroext %2, i16 zeroext %3) {
; CHECK-LABEL: select_cc_float_u16:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    fcmp.s %s0, %s0, %s1
; CHECK-NEXT:    cmov.s.eq %s3, %s2, %s0
; CHECK-NEXT:    adds.w.zx %s0, %s3, (0)1
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = fcmp fast oeq float %0, %1
  %6 = select i1 %5, i16 %2, i16 %3
  ret i16 %6
}

; Function Attrs: norecurse nounwind readnone
define zeroext i16 @select_cc_double_u16(double %0, double %1, i16 zeroext %2, i16 zeroext %3) {
; CHECK-LABEL: select_cc_double_u16:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    fcmp.d %s0, %s0, %s1
; CHECK-NEXT:    cmov.d.eq %s3, %s2, %s0
; CHECK-NEXT:    adds.w.zx %s0, %s3, (0)1
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = fcmp fast oeq double %0, %1
  %6 = select i1 %5, i16 %2, i16 %3
  ret i16 %6
}

; Function Attrs: norecurse nounwind readnone
define zeroext i16 @select_cc_quad_u16(fp128 %0, fp128 %1, i16 zeroext %2, i16 zeroext %3) {
; CHECK-LABEL: select_cc_quad_u16:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    fcmp.q %s0, %s0, %s2
; CHECK-NEXT:    cmov.d.eq %s5, %s4, %s0
; CHECK-NEXT:    adds.w.zx %s0, %s5, (0)1
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = fcmp fast oeq fp128 %0, %1
  %6 = select i1 %5, i16 %2, i16 %3
  ret i16 %6
}

; Function Attrs: norecurse nounwind readnone
define signext i32 @select_cc_i1_i32(i1 zeroext %0, i1 zeroext %1, i32 signext %2, i32 signext %3) {
; CHECK-LABEL: select_cc_i1_i32:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    xor %s0, %s0, %s1
; CHECK-NEXT:    cmov.w.ne %s2, %s3, %s0
; CHECK-NEXT:    adds.w.sx %s0, %s2, (0)1
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = xor i1 %0, %1
  %6 = select i1 %5, i32 %3, i32 %2
  ret i32 %6
}

; Function Attrs: norecurse nounwind readnone
define signext i32 @select_cc_i8_i32(i8 signext %0, i8 signext %1, i32 signext %2, i32 signext %3) {
; CHECK-LABEL: select_cc_i8_i32:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    cmps.w.sx %s0, %s0, %s1
; CHECK-NEXT:    cmov.w.eq %s3, %s2, %s0
; CHECK-NEXT:    adds.w.sx %s0, %s3, (0)1
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = icmp eq i8 %0, %1
  %6 = select i1 %5, i32 %2, i32 %3
  ret i32 %6
}

; Function Attrs: norecurse nounwind readnone
define signext i32 @select_cc_u8_i32(i8 zeroext %0, i8 zeroext %1, i32 signext %2, i32 signext %3) {
; CHECK-LABEL: select_cc_u8_i32:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    cmps.w.sx %s0, %s0, %s1
; CHECK-NEXT:    cmov.w.eq %s3, %s2, %s0
; CHECK-NEXT:    adds.w.sx %s0, %s3, (0)1
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = icmp eq i8 %0, %1
  %6 = select i1 %5, i32 %2, i32 %3
  ret i32 %6
}

; Function Attrs: norecurse nounwind readnone
define signext i32 @select_cc_i16_i32(i16 signext %0, i16 signext %1, i32 signext %2, i32 signext %3) {
; CHECK-LABEL: select_cc_i16_i32:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    cmps.w.sx %s0, %s0, %s1
; CHECK-NEXT:    cmov.w.eq %s3, %s2, %s0
; CHECK-NEXT:    adds.w.sx %s0, %s3, (0)1
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = icmp eq i16 %0, %1
  %6 = select i1 %5, i32 %2, i32 %3
  ret i32 %6
}

; Function Attrs: norecurse nounwind readnone
define signext i32 @select_cc_u16_i32(i16 zeroext %0, i16 zeroext %1, i32 signext %2, i32 signext %3) {
; CHECK-LABEL: select_cc_u16_i32:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    cmps.w.sx %s0, %s0, %s1
; CHECK-NEXT:    cmov.w.eq %s3, %s2, %s0
; CHECK-NEXT:    adds.w.sx %s0, %s3, (0)1
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = icmp eq i16 %0, %1
  %6 = select i1 %5, i32 %2, i32 %3
  ret i32 %6
}

; Function Attrs: norecurse nounwind readnone
define signext i32 @select_cc_i32_i32(i32 signext %0, i32 signext %1, i32 signext %2, i32 signext %3) {
; CHECK-LABEL: select_cc_i32_i32:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    cmps.w.sx %s0, %s0, %s1
; CHECK-NEXT:    cmov.w.eq %s3, %s2, %s0
; CHECK-NEXT:    adds.w.sx %s0, %s3, (0)1
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = icmp eq i32 %0, %1
  %6 = select i1 %5, i32 %2, i32 %3
  ret i32 %6
}

; Function Attrs: norecurse nounwind readnone
define signext i32 @select_cc_u32_i32(i32 zeroext %0, i32 zeroext %1, i32 signext %2, i32 signext %3) {
; CHECK-LABEL: select_cc_u32_i32:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    cmps.w.sx %s0, %s0, %s1
; CHECK-NEXT:    cmov.w.eq %s3, %s2, %s0
; CHECK-NEXT:    adds.w.sx %s0, %s3, (0)1
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = icmp eq i32 %0, %1
  %6 = select i1 %5, i32 %2, i32 %3
  ret i32 %6
}

; Function Attrs: norecurse nounwind readnone
define signext i32 @select_cc_i64_i32(i64 %0, i64 %1, i32 signext %2, i32 signext %3) {
; CHECK-LABEL: select_cc_i64_i32:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    cmps.l %s0, %s0, %s1
; CHECK-NEXT:    cmov.l.eq %s3, %s2, %s0
; CHECK-NEXT:    adds.w.sx %s0, %s3, (0)1
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = icmp eq i64 %0, %1
  %6 = select i1 %5, i32 %2, i32 %3
  ret i32 %6
}

; Function Attrs: norecurse nounwind readnone
define signext i32 @select_cc_u64_i32(i64 %0, i64 %1, i32 signext %2, i32 signext %3) {
; CHECK-LABEL: select_cc_u64_i32:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    cmps.l %s0, %s0, %s1
; CHECK-NEXT:    cmov.l.eq %s3, %s2, %s0
; CHECK-NEXT:    adds.w.sx %s0, %s3, (0)1
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = icmp eq i64 %0, %1
  %6 = select i1 %5, i32 %2, i32 %3
  ret i32 %6
}

; Function Attrs: norecurse nounwind readnone
define signext i32 @select_cc_i128_i32(i128 %0, i128 %1, i32 signext %2, i32 signext %3) {
; CHECK-LABEL: select_cc_i128_i32:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    xor %s1, %s1, %s3
; CHECK-NEXT:    xor %s0, %s0, %s2
; CHECK-NEXT:    or %s0, %s0, %s1
; CHECK-NEXT:    cmps.l %s0, %s0, (0)1
; CHECK-NEXT:    cmov.l.eq %s5, %s4, %s0
; CHECK-NEXT:    adds.w.sx %s0, %s5, (0)1
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = icmp eq i128 %0, %1
  %6 = select i1 %5, i32 %2, i32 %3
  ret i32 %6
}

; Function Attrs: norecurse nounwind readnone
define signext i32 @select_cc_u128_i32(i128 %0, i128 %1, i32 signext %2, i32 signext %3) {
; CHECK-LABEL: select_cc_u128_i32:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    xor %s1, %s1, %s3
; CHECK-NEXT:    xor %s0, %s0, %s2
; CHECK-NEXT:    or %s0, %s0, %s1
; CHECK-NEXT:    cmps.l %s0, %s0, (0)1
; CHECK-NEXT:    cmov.l.eq %s5, %s4, %s0
; CHECK-NEXT:    adds.w.sx %s0, %s5, (0)1
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = icmp eq i128 %0, %1
  %6 = select i1 %5, i32 %2, i32 %3
  ret i32 %6
}

; Function Attrs: norecurse nounwind readnone
define signext i32 @select_cc_float_i32(float %0, float %1, i32 signext %2, i32 signext %3) {
; CHECK-LABEL: select_cc_float_i32:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    fcmp.s %s0, %s0, %s1
; CHECK-NEXT:    cmov.s.eq %s3, %s2, %s0
; CHECK-NEXT:    adds.w.sx %s0, %s3, (0)1
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = fcmp fast oeq float %0, %1
  %6 = select i1 %5, i32 %2, i32 %3
  ret i32 %6
}

; Function Attrs: norecurse nounwind readnone
define signext i32 @select_cc_double_i32(double %0, double %1, i32 signext %2, i32 signext %3) {
; CHECK-LABEL: select_cc_double_i32:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    fcmp.d %s0, %s0, %s1
; CHECK-NEXT:    cmov.d.eq %s3, %s2, %s0
; CHECK-NEXT:    adds.w.sx %s0, %s3, (0)1
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = fcmp fast oeq double %0, %1
  %6 = select i1 %5, i32 %2, i32 %3
  ret i32 %6
}

; Function Attrs: norecurse nounwind readnone
define signext i32 @select_cc_quad_i32(fp128 %0, fp128 %1, i32 signext %2, i32 signext %3) {
; CHECK-LABEL: select_cc_quad_i32:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    fcmp.q %s0, %s0, %s2
; CHECK-NEXT:    cmov.d.eq %s5, %s4, %s0
; CHECK-NEXT:    adds.w.sx %s0, %s5, (0)1
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = fcmp fast oeq fp128 %0, %1
  %6 = select i1 %5, i32 %2, i32 %3
  ret i32 %6
}

; Function Attrs: norecurse nounwind readnone
define zeroext i32 @select_cc_i1_u32(i1 zeroext %0, i1 zeroext %1, i32 zeroext %2, i32 zeroext %3) {
; CHECK-LABEL: select_cc_i1_u32:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    xor %s0, %s0, %s1
; CHECK-NEXT:    cmov.w.ne %s2, %s3, %s0
; CHECK-NEXT:    adds.w.zx %s0, %s2, (0)1
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = xor i1 %0, %1
  %6 = select i1 %5, i32 %3, i32 %2
  ret i32 %6
}

; Function Attrs: norecurse nounwind readnone
define zeroext i32 @select_cc_i8_u32(i8 signext %0, i8 signext %1, i32 zeroext %2, i32 zeroext %3) {
; CHECK-LABEL: select_cc_i8_u32:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    cmps.w.sx %s0, %s0, %s1
; CHECK-NEXT:    cmov.w.eq %s3, %s2, %s0
; CHECK-NEXT:    adds.w.zx %s0, %s3, (0)1
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = icmp eq i8 %0, %1
  %6 = select i1 %5, i32 %2, i32 %3
  ret i32 %6
}

; Function Attrs: norecurse nounwind readnone
define zeroext i32 @select_cc_u8_u32(i8 zeroext %0, i8 zeroext %1, i32 zeroext %2, i32 zeroext %3) {
; CHECK-LABEL: select_cc_u8_u32:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    cmps.w.sx %s0, %s0, %s1
; CHECK-NEXT:    cmov.w.eq %s3, %s2, %s0
; CHECK-NEXT:    adds.w.zx %s0, %s3, (0)1
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = icmp eq i8 %0, %1
  %6 = select i1 %5, i32 %2, i32 %3
  ret i32 %6
}

; Function Attrs: norecurse nounwind readnone
define zeroext i32 @select_cc_i16_u32(i16 signext %0, i16 signext %1, i32 zeroext %2, i32 zeroext %3) {
; CHECK-LABEL: select_cc_i16_u32:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    cmps.w.sx %s0, %s0, %s1
; CHECK-NEXT:    cmov.w.eq %s3, %s2, %s0
; CHECK-NEXT:    adds.w.zx %s0, %s3, (0)1
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = icmp eq i16 %0, %1
  %6 = select i1 %5, i32 %2, i32 %3
  ret i32 %6
}

; Function Attrs: norecurse nounwind readnone
define zeroext i32 @select_cc_u16_u32(i16 zeroext %0, i16 zeroext %1, i32 zeroext %2, i32 zeroext %3) {
; CHECK-LABEL: select_cc_u16_u32:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    cmps.w.sx %s0, %s0, %s1
; CHECK-NEXT:    cmov.w.eq %s3, %s2, %s0
; CHECK-NEXT:    adds.w.zx %s0, %s3, (0)1
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = icmp eq i16 %0, %1
  %6 = select i1 %5, i32 %2, i32 %3
  ret i32 %6
}

; Function Attrs: norecurse nounwind readnone
define zeroext i32 @select_cc_i32_u32(i32 signext %0, i32 signext %1, i32 zeroext %2, i32 zeroext %3) {
; CHECK-LABEL: select_cc_i32_u32:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    cmps.w.sx %s0, %s0, %s1
; CHECK-NEXT:    cmov.w.eq %s3, %s2, %s0
; CHECK-NEXT:    adds.w.zx %s0, %s3, (0)1
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = icmp eq i32 %0, %1
  %6 = select i1 %5, i32 %2, i32 %3
  ret i32 %6
}

; Function Attrs: norecurse nounwind readnone
define zeroext i32 @select_cc_u32_u32(i32 zeroext %0, i32 zeroext %1, i32 zeroext %2, i32 zeroext %3) {
; CHECK-LABEL: select_cc_u32_u32:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    cmps.w.sx %s0, %s0, %s1
; CHECK-NEXT:    cmov.w.eq %s3, %s2, %s0
; CHECK-NEXT:    adds.w.zx %s0, %s3, (0)1
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = icmp eq i32 %0, %1
  %6 = select i1 %5, i32 %2, i32 %3
  ret i32 %6
}

; Function Attrs: norecurse nounwind readnone
define zeroext i32 @select_cc_i64_u32(i64 %0, i64 %1, i32 zeroext %2, i32 zeroext %3) {
; CHECK-LABEL: select_cc_i64_u32:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    cmps.l %s0, %s0, %s1
; CHECK-NEXT:    cmov.l.eq %s3, %s2, %s0
; CHECK-NEXT:    adds.w.zx %s0, %s3, (0)1
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = icmp eq i64 %0, %1
  %6 = select i1 %5, i32 %2, i32 %3
  ret i32 %6
}

; Function Attrs: norecurse nounwind readnone
define zeroext i32 @select_cc_u64_u32(i64 %0, i64 %1, i32 zeroext %2, i32 zeroext %3) {
; CHECK-LABEL: select_cc_u64_u32:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    cmps.l %s0, %s0, %s1
; CHECK-NEXT:    cmov.l.eq %s3, %s2, %s0
; CHECK-NEXT:    adds.w.zx %s0, %s3, (0)1
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = icmp eq i64 %0, %1
  %6 = select i1 %5, i32 %2, i32 %3
  ret i32 %6
}

; Function Attrs: norecurse nounwind readnone
define zeroext i32 @select_cc_i128_u32(i128 %0, i128 %1, i32 zeroext %2, i32 zeroext %3) {
; CHECK-LABEL: select_cc_i128_u32:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    xor %s1, %s1, %s3
; CHECK-NEXT:    xor %s0, %s0, %s2
; CHECK-NEXT:    or %s0, %s0, %s1
; CHECK-NEXT:    cmps.l %s0, %s0, (0)1
; CHECK-NEXT:    cmov.l.eq %s5, %s4, %s0
; CHECK-NEXT:    adds.w.zx %s0, %s5, (0)1
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = icmp eq i128 %0, %1
  %6 = select i1 %5, i32 %2, i32 %3
  ret i32 %6
}

; Function Attrs: norecurse nounwind readnone
define zeroext i32 @select_cc_u128_u32(i128 %0, i128 %1, i32 zeroext %2, i32 zeroext %3) {
; CHECK-LABEL: select_cc_u128_u32:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    xor %s1, %s1, %s3
; CHECK-NEXT:    xor %s0, %s0, %s2
; CHECK-NEXT:    or %s0, %s0, %s1
; CHECK-NEXT:    cmps.l %s0, %s0, (0)1
; CHECK-NEXT:    cmov.l.eq %s5, %s4, %s0
; CHECK-NEXT:    adds.w.zx %s0, %s5, (0)1
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = icmp eq i128 %0, %1
  %6 = select i1 %5, i32 %2, i32 %3
  ret i32 %6
}

; Function Attrs: norecurse nounwind readnone
define zeroext i32 @select_cc_float_u32(float %0, float %1, i32 zeroext %2, i32 zeroext %3) {
; CHECK-LABEL: select_cc_float_u32:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    fcmp.s %s0, %s0, %s1
; CHECK-NEXT:    cmov.s.eq %s3, %s2, %s0
; CHECK-NEXT:    adds.w.zx %s0, %s3, (0)1
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = fcmp fast oeq float %0, %1
  %6 = select i1 %5, i32 %2, i32 %3
  ret i32 %6
}

; Function Attrs: norecurse nounwind readnone
define zeroext i32 @select_cc_double_u32(double %0, double %1, i32 zeroext %2, i32 zeroext %3) {
; CHECK-LABEL: select_cc_double_u32:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    fcmp.d %s0, %s0, %s1
; CHECK-NEXT:    cmov.d.eq %s3, %s2, %s0
; CHECK-NEXT:    adds.w.zx %s0, %s3, (0)1
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = fcmp fast oeq double %0, %1
  %6 = select i1 %5, i32 %2, i32 %3
  ret i32 %6
}

; Function Attrs: norecurse nounwind readnone
define zeroext i32 @select_cc_quad_u32(fp128 %0, fp128 %1, i32 zeroext %2, i32 zeroext %3) {
; CHECK-LABEL: select_cc_quad_u32:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    fcmp.q %s0, %s0, %s2
; CHECK-NEXT:    cmov.d.eq %s5, %s4, %s0
; CHECK-NEXT:    adds.w.zx %s0, %s5, (0)1
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = fcmp fast oeq fp128 %0, %1
  %6 = select i1 %5, i32 %2, i32 %3
  ret i32 %6
}

; Function Attrs: norecurse nounwind readnone
define i64 @select_cc_i1_i64(i1 zeroext %0, i1 zeroext %1, i64 %2, i64 %3) {
; CHECK-LABEL: select_cc_i1_i64:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    xor %s0, %s0, %s1
; CHECK-NEXT:    adds.w.sx %s0, %s0, (0)1
; CHECK-NEXT:    cmov.w.ne %s2, %s3, %s0
; CHECK-NEXT:    or %s0, 0, %s2
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = xor i1 %0, %1
  %6 = select i1 %5, i64 %3, i64 %2
  ret i64 %6
}

; Function Attrs: norecurse nounwind readnone
define i64 @select_cc_i8_i64(i8 signext %0, i8 signext %1, i64 %2, i64 %3) {
; CHECK-LABEL: select_cc_i8_i64:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    cmps.w.sx %s0, %s0, %s1
; CHECK-NEXT:    cmov.w.eq %s3, %s2, %s0
; CHECK-NEXT:    or %s0, 0, %s3
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = icmp eq i8 %0, %1
  %6 = select i1 %5, i64 %2, i64 %3
  ret i64 %6
}

; Function Attrs: norecurse nounwind readnone
define i64 @select_cc_u8_i64(i8 zeroext %0, i8 zeroext %1, i64 %2, i64 %3) {
; CHECK-LABEL: select_cc_u8_i64:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    cmps.w.sx %s0, %s0, %s1
; CHECK-NEXT:    cmov.w.eq %s3, %s2, %s0
; CHECK-NEXT:    or %s0, 0, %s3
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = icmp eq i8 %0, %1
  %6 = select i1 %5, i64 %2, i64 %3
  ret i64 %6
}

; Function Attrs: norecurse nounwind readnone
define i64 @select_cc_i16_i64(i16 signext %0, i16 signext %1, i64 %2, i64 %3) {
; CHECK-LABEL: select_cc_i16_i64:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    cmps.w.sx %s0, %s0, %s1
; CHECK-NEXT:    cmov.w.eq %s3, %s2, %s0
; CHECK-NEXT:    or %s0, 0, %s3
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = icmp eq i16 %0, %1
  %6 = select i1 %5, i64 %2, i64 %3
  ret i64 %6
}

; Function Attrs: norecurse nounwind readnone
define i64 @select_cc_u16_i64(i16 zeroext %0, i16 zeroext %1, i64 %2, i64 %3) {
; CHECK-LABEL: select_cc_u16_i64:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    cmps.w.sx %s0, %s0, %s1
; CHECK-NEXT:    cmov.w.eq %s3, %s2, %s0
; CHECK-NEXT:    or %s0, 0, %s3
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = icmp eq i16 %0, %1
  %6 = select i1 %5, i64 %2, i64 %3
  ret i64 %6
}

; Function Attrs: norecurse nounwind readnone
define i64 @select_cc_i32_i64(i32 signext %0, i32 signext %1, i64 %2, i64 %3) {
; CHECK-LABEL: select_cc_i32_i64:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    cmps.w.sx %s0, %s0, %s1
; CHECK-NEXT:    cmov.w.eq %s3, %s2, %s0
; CHECK-NEXT:    or %s0, 0, %s3
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = icmp eq i32 %0, %1
  %6 = select i1 %5, i64 %2, i64 %3
  ret i64 %6
}

; Function Attrs: norecurse nounwind readnone
define i64 @select_cc_u32_i64(i32 zeroext %0, i32 zeroext %1, i64 %2, i64 %3) {
; CHECK-LABEL: select_cc_u32_i64:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    cmps.w.sx %s0, %s0, %s1
; CHECK-NEXT:    cmov.w.eq %s3, %s2, %s0
; CHECK-NEXT:    or %s0, 0, %s3
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = icmp eq i32 %0, %1
  %6 = select i1 %5, i64 %2, i64 %3
  ret i64 %6
}

; Function Attrs: norecurse nounwind readnone
define i64 @select_cc_i64_i64(i64 %0, i64 %1, i64 %2, i64 %3) {
; CHECK-LABEL: select_cc_i64_i64:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    cmps.l %s0, %s0, %s1
; CHECK-NEXT:    cmov.l.eq %s3, %s2, %s0
; CHECK-NEXT:    or %s0, 0, %s3
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = icmp eq i64 %0, %1
  %6 = select i1 %5, i64 %2, i64 %3
  ret i64 %6
}

; Function Attrs: norecurse nounwind readnone
define i64 @select_cc_u64_i64(i64 %0, i64 %1, i64 %2, i64 %3) {
; CHECK-LABEL: select_cc_u64_i64:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    cmps.l %s0, %s0, %s1
; CHECK-NEXT:    cmov.l.eq %s3, %s2, %s0
; CHECK-NEXT:    or %s0, 0, %s3
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = icmp eq i64 %0, %1
  %6 = select i1 %5, i64 %2, i64 %3
  ret i64 %6
}

; Function Attrs: norecurse nounwind readnone
define i64 @select_cc_i128_i64(i128 %0, i128 %1, i64 %2, i64 %3) {
; CHECK-LABEL: select_cc_i128_i64:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    xor %s1, %s1, %s3
; CHECK-NEXT:    xor %s0, %s0, %s2
; CHECK-NEXT:    or %s0, %s0, %s1
; CHECK-NEXT:    cmps.l %s0, %s0, (0)1
; CHECK-NEXT:    cmov.l.eq %s5, %s4, %s0
; CHECK-NEXT:    or %s0, 0, %s5
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = icmp eq i128 %0, %1
  %6 = select i1 %5, i64 %2, i64 %3
  ret i64 %6
}

; Function Attrs: norecurse nounwind readnone
define i64 @select_cc_u128_i64(i128 %0, i128 %1, i64 %2, i64 %3) {
; CHECK-LABEL: select_cc_u128_i64:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    xor %s1, %s1, %s3
; CHECK-NEXT:    xor %s0, %s0, %s2
; CHECK-NEXT:    or %s0, %s0, %s1
; CHECK-NEXT:    cmps.l %s0, %s0, (0)1
; CHECK-NEXT:    cmov.l.eq %s5, %s4, %s0
; CHECK-NEXT:    or %s0, 0, %s5
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = icmp eq i128 %0, %1
  %6 = select i1 %5, i64 %2, i64 %3
  ret i64 %6
}

; Function Attrs: norecurse nounwind readnone
define i64 @select_cc_float_i64(float %0, float %1, i64 %2, i64 %3) {
; CHECK-LABEL: select_cc_float_i64:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    fcmp.s %s0, %s0, %s1
; CHECK-NEXT:    cmov.s.eq %s3, %s2, %s0
; CHECK-NEXT:    or %s0, 0, %s3
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = fcmp fast oeq float %0, %1
  %6 = select i1 %5, i64 %2, i64 %3
  ret i64 %6
}

; Function Attrs: norecurse nounwind readnone
define i64 @select_cc_double_i64(double %0, double %1, i64 %2, i64 %3) {
; CHECK-LABEL: select_cc_double_i64:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    fcmp.d %s0, %s0, %s1
; CHECK-NEXT:    cmov.d.eq %s3, %s2, %s0
; CHECK-NEXT:    or %s0, 0, %s3
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = fcmp fast oeq double %0, %1
  %6 = select i1 %5, i64 %2, i64 %3
  ret i64 %6
}

; Function Attrs: norecurse nounwind readnone
define i64 @select_cc_quad_i64(fp128 %0, fp128 %1, i64 %2, i64 %3) {
; CHECK-LABEL: select_cc_quad_i64:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    fcmp.q %s0, %s0, %s2
; CHECK-NEXT:    cmov.d.eq %s5, %s4, %s0
; CHECK-NEXT:    or %s0, 0, %s5
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = fcmp fast oeq fp128 %0, %1
  %6 = select i1 %5, i64 %2, i64 %3
  ret i64 %6
}

; Function Attrs: norecurse nounwind readnone
define i64 @select_cc_i1_u64(i1 zeroext %0, i1 zeroext %1, i64 %2, i64 %3) {
; CHECK-LABEL: select_cc_i1_u64:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    xor %s0, %s0, %s1
; CHECK-NEXT:    adds.w.sx %s0, %s0, (0)1
; CHECK-NEXT:    cmov.w.ne %s2, %s3, %s0
; CHECK-NEXT:    or %s0, 0, %s2
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = xor i1 %0, %1
  %6 = select i1 %5, i64 %3, i64 %2
  ret i64 %6
}

; Function Attrs: norecurse nounwind readnone
define i64 @select_cc_i8_u64(i8 signext %0, i8 signext %1, i64 %2, i64 %3) {
; CHECK-LABEL: select_cc_i8_u64:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    cmps.w.sx %s0, %s0, %s1
; CHECK-NEXT:    cmov.w.eq %s3, %s2, %s0
; CHECK-NEXT:    or %s0, 0, %s3
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = icmp eq i8 %0, %1
  %6 = select i1 %5, i64 %2, i64 %3
  ret i64 %6
}

; Function Attrs: norecurse nounwind readnone
define i64 @select_cc_u8_u64(i8 zeroext %0, i8 zeroext %1, i64 %2, i64 %3) {
; CHECK-LABEL: select_cc_u8_u64:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    cmps.w.sx %s0, %s0, %s1
; CHECK-NEXT:    cmov.w.eq %s3, %s2, %s0
; CHECK-NEXT:    or %s0, 0, %s3
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = icmp eq i8 %0, %1
  %6 = select i1 %5, i64 %2, i64 %3
  ret i64 %6
}

; Function Attrs: norecurse nounwind readnone
define i64 @select_cc_i16_u64(i16 signext %0, i16 signext %1, i64 %2, i64 %3) {
; CHECK-LABEL: select_cc_i16_u64:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    cmps.w.sx %s0, %s0, %s1
; CHECK-NEXT:    cmov.w.eq %s3, %s2, %s0
; CHECK-NEXT:    or %s0, 0, %s3
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = icmp eq i16 %0, %1
  %6 = select i1 %5, i64 %2, i64 %3
  ret i64 %6
}

; Function Attrs: norecurse nounwind readnone
define i64 @select_cc_u16_u64(i16 zeroext %0, i16 zeroext %1, i64 %2, i64 %3) {
; CHECK-LABEL: select_cc_u16_u64:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    cmps.w.sx %s0, %s0, %s1
; CHECK-NEXT:    cmov.w.eq %s3, %s2, %s0
; CHECK-NEXT:    or %s0, 0, %s3
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = icmp eq i16 %0, %1
  %6 = select i1 %5, i64 %2, i64 %3
  ret i64 %6
}

; Function Attrs: norecurse nounwind readnone
define i64 @select_cc_i32_u64(i32 signext %0, i32 signext %1, i64 %2, i64 %3) {
; CHECK-LABEL: select_cc_i32_u64:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    cmps.w.sx %s0, %s0, %s1
; CHECK-NEXT:    cmov.w.eq %s3, %s2, %s0
; CHECK-NEXT:    or %s0, 0, %s3
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = icmp eq i32 %0, %1
  %6 = select i1 %5, i64 %2, i64 %3
  ret i64 %6
}

; Function Attrs: norecurse nounwind readnone
define i64 @select_cc_u32_u64(i32 zeroext %0, i32 zeroext %1, i64 %2, i64 %3) {
; CHECK-LABEL: select_cc_u32_u64:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    cmps.w.sx %s0, %s0, %s1
; CHECK-NEXT:    cmov.w.eq %s3, %s2, %s0
; CHECK-NEXT:    or %s0, 0, %s3
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = icmp eq i32 %0, %1
  %6 = select i1 %5, i64 %2, i64 %3
  ret i64 %6
}

; Function Attrs: norecurse nounwind readnone
define i64 @select_cc_i64_u64(i64 %0, i64 %1, i64 %2, i64 %3) {
; CHECK-LABEL: select_cc_i64_u64:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    cmps.l %s0, %s0, %s1
; CHECK-NEXT:    cmov.l.eq %s3, %s2, %s0
; CHECK-NEXT:    or %s0, 0, %s3
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = icmp eq i64 %0, %1
  %6 = select i1 %5, i64 %2, i64 %3
  ret i64 %6
}

; Function Attrs: norecurse nounwind readnone
define i64 @select_cc_u64_u64(i64 %0, i64 %1, i64 %2, i64 %3) {
; CHECK-LABEL: select_cc_u64_u64:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    cmps.l %s0, %s0, %s1
; CHECK-NEXT:    cmov.l.eq %s3, %s2, %s0
; CHECK-NEXT:    or %s0, 0, %s3
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = icmp eq i64 %0, %1
  %6 = select i1 %5, i64 %2, i64 %3
  ret i64 %6
}

; Function Attrs: norecurse nounwind readnone
define i64 @select_cc_i128_u64(i128 %0, i128 %1, i64 %2, i64 %3) {
; CHECK-LABEL: select_cc_i128_u64:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    xor %s1, %s1, %s3
; CHECK-NEXT:    xor %s0, %s0, %s2
; CHECK-NEXT:    or %s0, %s0, %s1
; CHECK-NEXT:    cmps.l %s0, %s0, (0)1
; CHECK-NEXT:    cmov.l.eq %s5, %s4, %s0
; CHECK-NEXT:    or %s0, 0, %s5
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = icmp eq i128 %0, %1
  %6 = select i1 %5, i64 %2, i64 %3
  ret i64 %6
}

; Function Attrs: norecurse nounwind readnone
define i64 @select_cc_u128_u64(i128 %0, i128 %1, i64 %2, i64 %3) {
; CHECK-LABEL: select_cc_u128_u64:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    xor %s1, %s1, %s3
; CHECK-NEXT:    xor %s0, %s0, %s2
; CHECK-NEXT:    or %s0, %s0, %s1
; CHECK-NEXT:    cmps.l %s0, %s0, (0)1
; CHECK-NEXT:    cmov.l.eq %s5, %s4, %s0
; CHECK-NEXT:    or %s0, 0, %s5
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = icmp eq i128 %0, %1
  %6 = select i1 %5, i64 %2, i64 %3
  ret i64 %6
}

; Function Attrs: norecurse nounwind readnone
define i64 @select_cc_float_u64(float %0, float %1, i64 %2, i64 %3) {
; CHECK-LABEL: select_cc_float_u64:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    fcmp.s %s0, %s0, %s1
; CHECK-NEXT:    cmov.s.eq %s3, %s2, %s0
; CHECK-NEXT:    or %s0, 0, %s3
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = fcmp fast oeq float %0, %1
  %6 = select i1 %5, i64 %2, i64 %3
  ret i64 %6
}

; Function Attrs: norecurse nounwind readnone
define i64 @select_cc_double_u64(double %0, double %1, i64 %2, i64 %3) {
; CHECK-LABEL: select_cc_double_u64:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    fcmp.d %s0, %s0, %s1
; CHECK-NEXT:    cmov.d.eq %s3, %s2, %s0
; CHECK-NEXT:    or %s0, 0, %s3
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = fcmp fast oeq double %0, %1
  %6 = select i1 %5, i64 %2, i64 %3
  ret i64 %6
}

; Function Attrs: norecurse nounwind readnone
define i64 @select_cc_quad_u64(fp128 %0, fp128 %1, i64 %2, i64 %3) {
; CHECK-LABEL: select_cc_quad_u64:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    fcmp.q %s0, %s0, %s2
; CHECK-NEXT:    cmov.d.eq %s5, %s4, %s0
; CHECK-NEXT:    or %s0, 0, %s5
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = fcmp fast oeq fp128 %0, %1
  %6 = select i1 %5, i64 %2, i64 %3
  ret i64 %6
}

; Function Attrs: norecurse nounwind readnone
define i128 @select_cc_i1_i128(i1 zeroext %0, i1 zeroext %1, i128 %2, i128 %3) {
; CHECK-LABEL: select_cc_i1_i128:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    xor %s0, %s0, %s1
; CHECK-NEXT:    adds.w.sx %s0, %s0, (0)1
; CHECK-NEXT:    cmov.w.ne %s2, %s4, %s0
; CHECK-NEXT:    cmov.w.ne %s3, %s5, %s0
; CHECK-NEXT:    or %s0, 0, %s2
; CHECK-NEXT:    or %s1, 0, %s3
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = xor i1 %0, %1
  %6 = select i1 %5, i128 %3, i128 %2
  ret i128 %6
}

; Function Attrs: norecurse nounwind readnone
define i128 @select_cc_i8_i128(i8 signext %0, i8 signext %1, i128 %2, i128 %3) {
; CHECK-LABEL: select_cc_i8_i128:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    cmps.w.sx %s0, %s0, %s1
; CHECK-NEXT:    cmov.w.eq %s4, %s2, %s0
; CHECK-NEXT:    cmov.w.eq %s5, %s3, %s0
; CHECK-NEXT:    or %s0, 0, %s4
; CHECK-NEXT:    or %s1, 0, %s5
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = icmp eq i8 %0, %1
  %6 = select i1 %5, i128 %2, i128 %3
  ret i128 %6
}

; Function Attrs: norecurse nounwind readnone
define i128 @select_cc_u8_i128(i8 zeroext %0, i8 zeroext %1, i128 %2, i128 %3) {
; CHECK-LABEL: select_cc_u8_i128:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    cmps.w.sx %s0, %s0, %s1
; CHECK-NEXT:    cmov.w.eq %s4, %s2, %s0
; CHECK-NEXT:    cmov.w.eq %s5, %s3, %s0
; CHECK-NEXT:    or %s0, 0, %s4
; CHECK-NEXT:    or %s1, 0, %s5
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = icmp eq i8 %0, %1
  %6 = select i1 %5, i128 %2, i128 %3
  ret i128 %6
}

; Function Attrs: norecurse nounwind readnone
define i128 @select_cc_i16_i128(i16 signext %0, i16 signext %1, i128 %2, i128 %3) {
; CHECK-LABEL: select_cc_i16_i128:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    cmps.w.sx %s0, %s0, %s1
; CHECK-NEXT:    cmov.w.eq %s4, %s2, %s0
; CHECK-NEXT:    cmov.w.eq %s5, %s3, %s0
; CHECK-NEXT:    or %s0, 0, %s4
; CHECK-NEXT:    or %s1, 0, %s5
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = icmp eq i16 %0, %1
  %6 = select i1 %5, i128 %2, i128 %3
  ret i128 %6
}

; Function Attrs: norecurse nounwind readnone
define i128 @select_cc_u16_i128(i16 zeroext %0, i16 zeroext %1, i128 %2, i128 %3) {
; CHECK-LABEL: select_cc_u16_i128:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    cmps.w.sx %s0, %s0, %s1
; CHECK-NEXT:    cmov.w.eq %s4, %s2, %s0
; CHECK-NEXT:    cmov.w.eq %s5, %s3, %s0
; CHECK-NEXT:    or %s0, 0, %s4
; CHECK-NEXT:    or %s1, 0, %s5
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = icmp eq i16 %0, %1
  %6 = select i1 %5, i128 %2, i128 %3
  ret i128 %6
}

; Function Attrs: norecurse nounwind readnone
define i128 @select_cc_i32_i128(i32 signext %0, i32 signext %1, i128 %2, i128 %3) {
; CHECK-LABEL: select_cc_i32_i128:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    cmps.w.sx %s0, %s0, %s1
; CHECK-NEXT:    cmov.w.eq %s4, %s2, %s0
; CHECK-NEXT:    cmov.w.eq %s5, %s3, %s0
; CHECK-NEXT:    or %s0, 0, %s4
; CHECK-NEXT:    or %s1, 0, %s5
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = icmp eq i32 %0, %1
  %6 = select i1 %5, i128 %2, i128 %3
  ret i128 %6
}

; Function Attrs: norecurse nounwind readnone
define i128 @select_cc_u32_i128(i32 zeroext %0, i32 zeroext %1, i128 %2, i128 %3) {
; CHECK-LABEL: select_cc_u32_i128:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    cmps.w.sx %s0, %s0, %s1
; CHECK-NEXT:    cmov.w.eq %s4, %s2, %s0
; CHECK-NEXT:    cmov.w.eq %s5, %s3, %s0
; CHECK-NEXT:    or %s0, 0, %s4
; CHECK-NEXT:    or %s1, 0, %s5
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = icmp eq i32 %0, %1
  %6 = select i1 %5, i128 %2, i128 %3
  ret i128 %6
}

; Function Attrs: norecurse nounwind readnone
define i128 @select_cc_i64_i128(i64 %0, i64 %1, i128 %2, i128 %3) {
; CHECK-LABEL: select_cc_i64_i128:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    cmps.l %s0, %s0, %s1
; CHECK-NEXT:    cmov.l.eq %s4, %s2, %s0
; CHECK-NEXT:    cmov.l.eq %s5, %s3, %s0
; CHECK-NEXT:    or %s0, 0, %s4
; CHECK-NEXT:    or %s1, 0, %s5
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = icmp eq i64 %0, %1
  %6 = select i1 %5, i128 %2, i128 %3
  ret i128 %6
}

; Function Attrs: norecurse nounwind readnone
define i128 @select_cc_u64_i128(i64 %0, i64 %1, i128 %2, i128 %3) {
; CHECK-LABEL: select_cc_u64_i128:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    cmps.l %s0, %s0, %s1
; CHECK-NEXT:    cmov.l.eq %s4, %s2, %s0
; CHECK-NEXT:    cmov.l.eq %s5, %s3, %s0
; CHECK-NEXT:    or %s0, 0, %s4
; CHECK-NEXT:    or %s1, 0, %s5
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = icmp eq i64 %0, %1
  %6 = select i1 %5, i128 %2, i128 %3
  ret i128 %6
}

; Function Attrs: norecurse nounwind readnone
define i128 @select_cc_i128_i128(i128 %0, i128 %1, i128 %2, i128 %3) {
; CHECK-LABEL: select_cc_i128_i128:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    xor %s1, %s1, %s3
; CHECK-NEXT:    xor %s0, %s0, %s2
; CHECK-NEXT:    or %s0, %s0, %s1
; CHECK-NEXT:    cmps.l %s0, %s0, (0)1
; CHECK-NEXT:    cmov.l.eq %s6, %s4, %s0
; CHECK-NEXT:    cmov.l.eq %s7, %s5, %s0
; CHECK-NEXT:    or %s0, 0, %s6
; CHECK-NEXT:    or %s1, 0, %s7
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = icmp eq i128 %0, %1
  %6 = select i1 %5, i128 %2, i128 %3
  ret i128 %6
}

; Function Attrs: norecurse nounwind readnone
define i128 @select_cc_u128_i128(i128 %0, i128 %1, i128 %2, i128 %3) {
; CHECK-LABEL: select_cc_u128_i128:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    xor %s1, %s1, %s3
; CHECK-NEXT:    xor %s0, %s0, %s2
; CHECK-NEXT:    or %s0, %s0, %s1
; CHECK-NEXT:    cmps.l %s0, %s0, (0)1
; CHECK-NEXT:    cmov.l.eq %s6, %s4, %s0
; CHECK-NEXT:    cmov.l.eq %s7, %s5, %s0
; CHECK-NEXT:    or %s0, 0, %s6
; CHECK-NEXT:    or %s1, 0, %s7
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = icmp eq i128 %0, %1
  %6 = select i1 %5, i128 %2, i128 %3
  ret i128 %6
}

; Function Attrs: norecurse nounwind readnone
define i128 @select_cc_float_i128(float %0, float %1, i128 %2, i128 %3) {
; CHECK-LABEL: select_cc_float_i128:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    fcmp.s %s0, %s0, %s1
; CHECK-NEXT:    cmov.s.eq %s4, %s2, %s0
; CHECK-NEXT:    cmov.s.eq %s5, %s3, %s0
; CHECK-NEXT:    or %s0, 0, %s4
; CHECK-NEXT:    or %s1, 0, %s5
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = fcmp fast oeq float %0, %1
  %6 = select i1 %5, i128 %2, i128 %3
  ret i128 %6
}

; Function Attrs: norecurse nounwind readnone
define i128 @select_cc_double_i128(double %0, double %1, i128 %2, i128 %3) {
; CHECK-LABEL: select_cc_double_i128:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    fcmp.d %s0, %s0, %s1
; CHECK-NEXT:    cmov.d.eq %s4, %s2, %s0
; CHECK-NEXT:    cmov.d.eq %s5, %s3, %s0
; CHECK-NEXT:    or %s0, 0, %s4
; CHECK-NEXT:    or %s1, 0, %s5
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = fcmp fast oeq double %0, %1
  %6 = select i1 %5, i128 %2, i128 %3
  ret i128 %6
}

; Function Attrs: norecurse nounwind readnone
define i128 @select_cc_quad_i128(fp128 %0, fp128 %1, i128 %2, i128 %3) {
; CHECK-LABEL: select_cc_quad_i128:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    fcmp.q %s0, %s0, %s2
; CHECK-NEXT:    cmov.d.eq %s6, %s4, %s0
; CHECK-NEXT:    cmov.d.eq %s7, %s5, %s0
; CHECK-NEXT:    or %s0, 0, %s6
; CHECK-NEXT:    or %s1, 0, %s7
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = fcmp fast oeq fp128 %0, %1
  %6 = select i1 %5, i128 %2, i128 %3
  ret i128 %6
}

; Function Attrs: norecurse nounwind readnone
define i128 @select_cc_i1_u128(i1 zeroext %0, i1 zeroext %1, i128 %2, i128 %3) {
; CHECK-LABEL: select_cc_i1_u128:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    xor %s0, %s0, %s1
; CHECK-NEXT:    adds.w.sx %s0, %s0, (0)1
; CHECK-NEXT:    cmov.w.ne %s2, %s4, %s0
; CHECK-NEXT:    cmov.w.ne %s3, %s5, %s0
; CHECK-NEXT:    or %s0, 0, %s2
; CHECK-NEXT:    or %s1, 0, %s3
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = xor i1 %0, %1
  %6 = select i1 %5, i128 %3, i128 %2
  ret i128 %6
}

; Function Attrs: norecurse nounwind readnone
define i128 @select_cc_i8_u128(i8 signext %0, i8 signext %1, i128 %2, i128 %3) {
; CHECK-LABEL: select_cc_i8_u128:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    cmps.w.sx %s0, %s0, %s1
; CHECK-NEXT:    cmov.w.eq %s4, %s2, %s0
; CHECK-NEXT:    cmov.w.eq %s5, %s3, %s0
; CHECK-NEXT:    or %s0, 0, %s4
; CHECK-NEXT:    or %s1, 0, %s5
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = icmp eq i8 %0, %1
  %6 = select i1 %5, i128 %2, i128 %3
  ret i128 %6
}

; Function Attrs: norecurse nounwind readnone
define i128 @select_cc_u8_u128(i8 zeroext %0, i8 zeroext %1, i128 %2, i128 %3) {
; CHECK-LABEL: select_cc_u8_u128:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    cmps.w.sx %s0, %s0, %s1
; CHECK-NEXT:    cmov.w.eq %s4, %s2, %s0
; CHECK-NEXT:    cmov.w.eq %s5, %s3, %s0
; CHECK-NEXT:    or %s0, 0, %s4
; CHECK-NEXT:    or %s1, 0, %s5
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = icmp eq i8 %0, %1
  %6 = select i1 %5, i128 %2, i128 %3
  ret i128 %6
}

; Function Attrs: norecurse nounwind readnone
define i128 @select_cc_i16_u128(i16 signext %0, i16 signext %1, i128 %2, i128 %3) {
; CHECK-LABEL: select_cc_i16_u128:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    cmps.w.sx %s0, %s0, %s1
; CHECK-NEXT:    cmov.w.eq %s4, %s2, %s0
; CHECK-NEXT:    cmov.w.eq %s5, %s3, %s0
; CHECK-NEXT:    or %s0, 0, %s4
; CHECK-NEXT:    or %s1, 0, %s5
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = icmp eq i16 %0, %1
  %6 = select i1 %5, i128 %2, i128 %3
  ret i128 %6
}

; Function Attrs: norecurse nounwind readnone
define i128 @select_cc_u16_u128(i16 zeroext %0, i16 zeroext %1, i128 %2, i128 %3) {
; CHECK-LABEL: select_cc_u16_u128:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    cmps.w.sx %s0, %s0, %s1
; CHECK-NEXT:    cmov.w.eq %s4, %s2, %s0
; CHECK-NEXT:    cmov.w.eq %s5, %s3, %s0
; CHECK-NEXT:    or %s0, 0, %s4
; CHECK-NEXT:    or %s1, 0, %s5
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = icmp eq i16 %0, %1
  %6 = select i1 %5, i128 %2, i128 %3
  ret i128 %6
}

; Function Attrs: norecurse nounwind readnone
define i128 @select_cc_i32_u128(i32 signext %0, i32 signext %1, i128 %2, i128 %3) {
; CHECK-LABEL: select_cc_i32_u128:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    cmps.w.sx %s0, %s0, %s1
; CHECK-NEXT:    cmov.w.eq %s4, %s2, %s0
; CHECK-NEXT:    cmov.w.eq %s5, %s3, %s0
; CHECK-NEXT:    or %s0, 0, %s4
; CHECK-NEXT:    or %s1, 0, %s5
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = icmp eq i32 %0, %1
  %6 = select i1 %5, i128 %2, i128 %3
  ret i128 %6
}

; Function Attrs: norecurse nounwind readnone
define i128 @select_cc_u32_u128(i32 zeroext %0, i32 zeroext %1, i128 %2, i128 %3) {
; CHECK-LABEL: select_cc_u32_u128:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    cmps.w.sx %s0, %s0, %s1
; CHECK-NEXT:    cmov.w.eq %s4, %s2, %s0
; CHECK-NEXT:    cmov.w.eq %s5, %s3, %s0
; CHECK-NEXT:    or %s0, 0, %s4
; CHECK-NEXT:    or %s1, 0, %s5
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = icmp eq i32 %0, %1
  %6 = select i1 %5, i128 %2, i128 %3
  ret i128 %6
}

; Function Attrs: norecurse nounwind readnone
define i128 @select_cc_i64_u128(i64 %0, i64 %1, i128 %2, i128 %3) {
; CHECK-LABEL: select_cc_i64_u128:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    cmps.l %s0, %s0, %s1
; CHECK-NEXT:    cmov.l.eq %s4, %s2, %s0
; CHECK-NEXT:    cmov.l.eq %s5, %s3, %s0
; CHECK-NEXT:    or %s0, 0, %s4
; CHECK-NEXT:    or %s1, 0, %s5
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = icmp eq i64 %0, %1
  %6 = select i1 %5, i128 %2, i128 %3
  ret i128 %6
}

; Function Attrs: norecurse nounwind readnone
define i128 @select_cc_u64_u128(i64 %0, i64 %1, i128 %2, i128 %3) {
; CHECK-LABEL: select_cc_u64_u128:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    cmps.l %s0, %s0, %s1
; CHECK-NEXT:    cmov.l.eq %s4, %s2, %s0
; CHECK-NEXT:    cmov.l.eq %s5, %s3, %s0
; CHECK-NEXT:    or %s0, 0, %s4
; CHECK-NEXT:    or %s1, 0, %s5
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = icmp eq i64 %0, %1
  %6 = select i1 %5, i128 %2, i128 %3
  ret i128 %6
}

; Function Attrs: norecurse nounwind readnone
define i128 @select_cc_i128_u128(i128 %0, i128 %1, i128 %2, i128 %3) {
; CHECK-LABEL: select_cc_i128_u128:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    xor %s1, %s1, %s3
; CHECK-NEXT:    xor %s0, %s0, %s2
; CHECK-NEXT:    or %s0, %s0, %s1
; CHECK-NEXT:    cmps.l %s0, %s0, (0)1
; CHECK-NEXT:    cmov.l.eq %s6, %s4, %s0
; CHECK-NEXT:    cmov.l.eq %s7, %s5, %s0
; CHECK-NEXT:    or %s0, 0, %s6
; CHECK-NEXT:    or %s1, 0, %s7
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = icmp eq i128 %0, %1
  %6 = select i1 %5, i128 %2, i128 %3
  ret i128 %6
}

; Function Attrs: norecurse nounwind readnone
define i128 @select_cc_u128_u128(i128 %0, i128 %1, i128 %2, i128 %3) {
; CHECK-LABEL: select_cc_u128_u128:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    xor %s1, %s1, %s3
; CHECK-NEXT:    xor %s0, %s0, %s2
; CHECK-NEXT:    or %s0, %s0, %s1
; CHECK-NEXT:    cmps.l %s0, %s0, (0)1
; CHECK-NEXT:    cmov.l.eq %s6, %s4, %s0
; CHECK-NEXT:    cmov.l.eq %s7, %s5, %s0
; CHECK-NEXT:    or %s0, 0, %s6
; CHECK-NEXT:    or %s1, 0, %s7
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = icmp eq i128 %0, %1
  %6 = select i1 %5, i128 %2, i128 %3
  ret i128 %6
}

; Function Attrs: norecurse nounwind readnone
define i128 @select_cc_float_u128(float %0, float %1, i128 %2, i128 %3) {
; CHECK-LABEL: select_cc_float_u128:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    fcmp.s %s0, %s0, %s1
; CHECK-NEXT:    cmov.s.eq %s4, %s2, %s0
; CHECK-NEXT:    cmov.s.eq %s5, %s3, %s0
; CHECK-NEXT:    or %s0, 0, %s4
; CHECK-NEXT:    or %s1, 0, %s5
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = fcmp fast oeq float %0, %1
  %6 = select i1 %5, i128 %2, i128 %3
  ret i128 %6
}

; Function Attrs: norecurse nounwind readnone
define i128 @select_cc_double_u128(double %0, double %1, i128 %2, i128 %3) {
; CHECK-LABEL: select_cc_double_u128:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    fcmp.d %s0, %s0, %s1
; CHECK-NEXT:    cmov.d.eq %s4, %s2, %s0
; CHECK-NEXT:    cmov.d.eq %s5, %s3, %s0
; CHECK-NEXT:    or %s0, 0, %s4
; CHECK-NEXT:    or %s1, 0, %s5
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = fcmp fast oeq double %0, %1
  %6 = select i1 %5, i128 %2, i128 %3
  ret i128 %6
}

; Function Attrs: norecurse nounwind readnone
define i128 @select_cc_quad_u128(fp128 %0, fp128 %1, i128 %2, i128 %3) {
; CHECK-LABEL: select_cc_quad_u128:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    fcmp.q %s0, %s0, %s2
; CHECK-NEXT:    cmov.d.eq %s6, %s4, %s0
; CHECK-NEXT:    cmov.d.eq %s7, %s5, %s0
; CHECK-NEXT:    or %s0, 0, %s6
; CHECK-NEXT:    or %s1, 0, %s7
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = fcmp fast oeq fp128 %0, %1
  %6 = select i1 %5, i128 %2, i128 %3
  ret i128 %6
}

; Function Attrs: norecurse nounwind readnone
define float @select_cc_i1_float(i1 zeroext %0, i1 zeroext %1, float %2, float %3) {
; CHECK-LABEL: select_cc_i1_float:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    xor %s0, %s0, %s1
; CHECK-NEXT:    adds.w.sx %s0, %s0, (0)1
; CHECK-NEXT:    cmov.w.ne %s2, %s3, %s0
; CHECK-NEXT:    or %s0, 0, %s2
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = xor i1 %0, %1
  %6 = select fast i1 %5, float %3, float %2
  ret float %6
}

; Function Attrs: norecurse nounwind readnone
define float @select_cc_i8_float(i8 signext %0, i8 signext %1, float %2, float %3) {
; CHECK-LABEL: select_cc_i8_float:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    cmps.w.sx %s0, %s0, %s1
; CHECK-NEXT:    cmov.w.eq %s3, %s2, %s0
; CHECK-NEXT:    or %s0, 0, %s3
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = icmp eq i8 %0, %1
  %6 = select fast i1 %5, float %2, float %3
  ret float %6
}

; Function Attrs: norecurse nounwind readnone
define float @select_cc_u8_float(i8 zeroext %0, i8 zeroext %1, float %2, float %3) {
; CHECK-LABEL: select_cc_u8_float:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    cmps.w.sx %s0, %s0, %s1
; CHECK-NEXT:    cmov.w.eq %s3, %s2, %s0
; CHECK-NEXT:    or %s0, 0, %s3
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = icmp eq i8 %0, %1
  %6 = select fast i1 %5, float %2, float %3
  ret float %6
}

; Function Attrs: norecurse nounwind readnone
define float @select_cc_i16_float(i16 signext %0, i16 signext %1, float %2, float %3) {
; CHECK-LABEL: select_cc_i16_float:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    cmps.w.sx %s0, %s0, %s1
; CHECK-NEXT:    cmov.w.eq %s3, %s2, %s0
; CHECK-NEXT:    or %s0, 0, %s3
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = icmp eq i16 %0, %1
  %6 = select fast i1 %5, float %2, float %3
  ret float %6
}

; Function Attrs: norecurse nounwind readnone
define float @select_cc_u16_float(i16 zeroext %0, i16 zeroext %1, float %2, float %3) {
; CHECK-LABEL: select_cc_u16_float:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    cmps.w.sx %s0, %s0, %s1
; CHECK-NEXT:    cmov.w.eq %s3, %s2, %s0
; CHECK-NEXT:    or %s0, 0, %s3
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = icmp eq i16 %0, %1
  %6 = select fast i1 %5, float %2, float %3
  ret float %6
}

; Function Attrs: norecurse nounwind readnone
define float @select_cc_i32_float(i32 signext %0, i32 signext %1, float %2, float %3) {
; CHECK-LABEL: select_cc_i32_float:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    cmps.w.sx %s0, %s0, %s1
; CHECK-NEXT:    cmov.w.eq %s3, %s2, %s0
; CHECK-NEXT:    or %s0, 0, %s3
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = icmp eq i32 %0, %1
  %6 = select fast i1 %5, float %2, float %3
  ret float %6
}

; Function Attrs: norecurse nounwind readnone
define float @select_cc_u32_float(i32 zeroext %0, i32 zeroext %1, float %2, float %3) {
; CHECK-LABEL: select_cc_u32_float:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    cmps.w.sx %s0, %s0, %s1
; CHECK-NEXT:    cmov.w.eq %s3, %s2, %s0
; CHECK-NEXT:    or %s0, 0, %s3
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = icmp eq i32 %0, %1
  %6 = select fast i1 %5, float %2, float %3
  ret float %6
}

; Function Attrs: norecurse nounwind readnone
define float @select_cc_i64_float(i64 %0, i64 %1, float %2, float %3) {
; CHECK-LABEL: select_cc_i64_float:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    cmps.l %s0, %s0, %s1
; CHECK-NEXT:    cmov.l.eq %s3, %s2, %s0
; CHECK-NEXT:    or %s0, 0, %s3
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = icmp eq i64 %0, %1
  %6 = select fast i1 %5, float %2, float %3
  ret float %6
}

; Function Attrs: norecurse nounwind readnone
define float @select_cc_u64_float(i64 %0, i64 %1, float %2, float %3) {
; CHECK-LABEL: select_cc_u64_float:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    cmps.l %s0, %s0, %s1
; CHECK-NEXT:    cmov.l.eq %s3, %s2, %s0
; CHECK-NEXT:    or %s0, 0, %s3
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = icmp eq i64 %0, %1
  %6 = select fast i1 %5, float %2, float %3
  ret float %6
}

; Function Attrs: norecurse nounwind readnone
define float @select_cc_i128_float(i128 %0, i128 %1, float %2, float %3) {
; CHECK-LABEL: select_cc_i128_float:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    xor %s1, %s1, %s3
; CHECK-NEXT:    xor %s0, %s0, %s2
; CHECK-NEXT:    or %s0, %s0, %s1
; CHECK-NEXT:    cmps.l %s0, %s0, (0)1
; CHECK-NEXT:    cmov.l.eq %s5, %s4, %s0
; CHECK-NEXT:    or %s0, 0, %s5
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = icmp eq i128 %0, %1
  %6 = select fast i1 %5, float %2, float %3
  ret float %6
}

; Function Attrs: norecurse nounwind readnone
define float @select_cc_u128_float(i128 %0, i128 %1, float %2, float %3) {
; CHECK-LABEL: select_cc_u128_float:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    xor %s1, %s1, %s3
; CHECK-NEXT:    xor %s0, %s0, %s2
; CHECK-NEXT:    or %s0, %s0, %s1
; CHECK-NEXT:    cmps.l %s0, %s0, (0)1
; CHECK-NEXT:    cmov.l.eq %s5, %s4, %s0
; CHECK-NEXT:    or %s0, 0, %s5
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = icmp eq i128 %0, %1
  %6 = select fast i1 %5, float %2, float %3
  ret float %6
}

; Function Attrs: norecurse nounwind readnone
define float @select_cc_float_float(float %0, float %1, float %2, float %3) {
; CHECK-LABEL: select_cc_float_float:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    fcmp.s %s0, %s0, %s1
; CHECK-NEXT:    cmov.s.eq %s3, %s2, %s0
; CHECK-NEXT:    or %s0, 0, %s3
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = fcmp fast oeq float %0, %1
  %6 = select fast i1 %5, float %2, float %3
  ret float %6
}

; Function Attrs: norecurse nounwind readnone
define float @select_cc_double_float(double %0, double %1, float %2, float %3) {
; CHECK-LABEL: select_cc_double_float:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    fcmp.d %s0, %s0, %s1
; CHECK-NEXT:    cmov.d.eq %s3, %s2, %s0
; CHECK-NEXT:    or %s0, 0, %s3
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = fcmp fast oeq double %0, %1
  %6 = select fast i1 %5, float %2, float %3
  ret float %6
}

; Function Attrs: norecurse nounwind readnone
define float @select_cc_quad_float(fp128 %0, fp128 %1, float %2, float %3) {
; CHECK-LABEL: select_cc_quad_float:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    fcmp.q %s0, %s0, %s2
; CHECK-NEXT:    cmov.d.eq %s5, %s4, %s0
; CHECK-NEXT:    or %s0, 0, %s5
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = fcmp fast oeq fp128 %0, %1
  %6 = select fast i1 %5, float %2, float %3
  ret float %6
}

; Function Attrs: norecurse nounwind readnone
define double @select_cc_i1_double(i1 zeroext %0, i1 zeroext %1, double %2, double %3) {
; CHECK-LABEL: select_cc_i1_double:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    xor %s0, %s0, %s1
; CHECK-NEXT:    adds.w.sx %s0, %s0, (0)1
; CHECK-NEXT:    cmov.w.ne %s2, %s3, %s0
; CHECK-NEXT:    or %s0, 0, %s2
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = xor i1 %0, %1
  %6 = select fast i1 %5, double %3, double %2
  ret double %6
}

; Function Attrs: norecurse nounwind readnone
define double @select_cc_i8_double(i8 signext %0, i8 signext %1, double %2, double %3) {
; CHECK-LABEL: select_cc_i8_double:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    cmps.w.sx %s0, %s0, %s1
; CHECK-NEXT:    cmov.w.eq %s3, %s2, %s0
; CHECK-NEXT:    or %s0, 0, %s3
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = icmp eq i8 %0, %1
  %6 = select fast i1 %5, double %2, double %3
  ret double %6
}

; Function Attrs: norecurse nounwind readnone
define double @select_cc_u8_double(i8 zeroext %0, i8 zeroext %1, double %2, double %3) {
; CHECK-LABEL: select_cc_u8_double:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    cmps.w.sx %s0, %s0, %s1
; CHECK-NEXT:    cmov.w.eq %s3, %s2, %s0
; CHECK-NEXT:    or %s0, 0, %s3
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = icmp eq i8 %0, %1
  %6 = select fast i1 %5, double %2, double %3
  ret double %6
}

; Function Attrs: norecurse nounwind readnone
define double @select_cc_i16_double(i16 signext %0, i16 signext %1, double %2, double %3) {
; CHECK-LABEL: select_cc_i16_double:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    cmps.w.sx %s0, %s0, %s1
; CHECK-NEXT:    cmov.w.eq %s3, %s2, %s0
; CHECK-NEXT:    or %s0, 0, %s3
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = icmp eq i16 %0, %1
  %6 = select fast i1 %5, double %2, double %3
  ret double %6
}

; Function Attrs: norecurse nounwind readnone
define double @select_cc_u16_double(i16 zeroext %0, i16 zeroext %1, double %2, double %3) {
; CHECK-LABEL: select_cc_u16_double:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    cmps.w.sx %s0, %s0, %s1
; CHECK-NEXT:    cmov.w.eq %s3, %s2, %s0
; CHECK-NEXT:    or %s0, 0, %s3
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = icmp eq i16 %0, %1
  %6 = select fast i1 %5, double %2, double %3
  ret double %6
}

; Function Attrs: norecurse nounwind readnone
define double @select_cc_i32_double(i32 signext %0, i32 signext %1, double %2, double %3) {
; CHECK-LABEL: select_cc_i32_double:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    cmps.w.sx %s0, %s0, %s1
; CHECK-NEXT:    cmov.w.eq %s3, %s2, %s0
; CHECK-NEXT:    or %s0, 0, %s3
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = icmp eq i32 %0, %1
  %6 = select fast i1 %5, double %2, double %3
  ret double %6
}

; Function Attrs: norecurse nounwind readnone
define double @select_cc_u32_double(i32 zeroext %0, i32 zeroext %1, double %2, double %3) {
; CHECK-LABEL: select_cc_u32_double:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    cmps.w.sx %s0, %s0, %s1
; CHECK-NEXT:    cmov.w.eq %s3, %s2, %s0
; CHECK-NEXT:    or %s0, 0, %s3
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = icmp eq i32 %0, %1
  %6 = select fast i1 %5, double %2, double %3
  ret double %6
}

; Function Attrs: norecurse nounwind readnone
define double @select_cc_i64_double(i64 %0, i64 %1, double %2, double %3) {
; CHECK-LABEL: select_cc_i64_double:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    cmps.l %s0, %s0, %s1
; CHECK-NEXT:    cmov.l.eq %s3, %s2, %s0
; CHECK-NEXT:    or %s0, 0, %s3
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = icmp eq i64 %0, %1
  %6 = select fast i1 %5, double %2, double %3
  ret double %6
}

; Function Attrs: norecurse nounwind readnone
define double @select_cc_u64_double(i64 %0, i64 %1, double %2, double %3) {
; CHECK-LABEL: select_cc_u64_double:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    cmps.l %s0, %s0, %s1
; CHECK-NEXT:    cmov.l.eq %s3, %s2, %s0
; CHECK-NEXT:    or %s0, 0, %s3
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = icmp eq i64 %0, %1
  %6 = select fast i1 %5, double %2, double %3
  ret double %6
}

; Function Attrs: norecurse nounwind readnone
define double @select_cc_i128_double(i128 %0, i128 %1, double %2, double %3) {
; CHECK-LABEL: select_cc_i128_double:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    xor %s1, %s1, %s3
; CHECK-NEXT:    xor %s0, %s0, %s2
; CHECK-NEXT:    or %s0, %s0, %s1
; CHECK-NEXT:    cmps.l %s0, %s0, (0)1
; CHECK-NEXT:    cmov.l.eq %s5, %s4, %s0
; CHECK-NEXT:    or %s0, 0, %s5
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = icmp eq i128 %0, %1
  %6 = select fast i1 %5, double %2, double %3
  ret double %6
}

; Function Attrs: norecurse nounwind readnone
define double @select_cc_u128_double(i128 %0, i128 %1, double %2, double %3) {
; CHECK-LABEL: select_cc_u128_double:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    xor %s1, %s1, %s3
; CHECK-NEXT:    xor %s0, %s0, %s2
; CHECK-NEXT:    or %s0, %s0, %s1
; CHECK-NEXT:    cmps.l %s0, %s0, (0)1
; CHECK-NEXT:    cmov.l.eq %s5, %s4, %s0
; CHECK-NEXT:    or %s0, 0, %s5
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = icmp eq i128 %0, %1
  %6 = select fast i1 %5, double %2, double %3
  ret double %6
}

; Function Attrs: norecurse nounwind readnone
define double @select_cc_float_double(float %0, float %1, double %2, double %3) {
; CHECK-LABEL: select_cc_float_double:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    fcmp.s %s0, %s0, %s1
; CHECK-NEXT:    cmov.s.eq %s3, %s2, %s0
; CHECK-NEXT:    or %s0, 0, %s3
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = fcmp fast oeq float %0, %1
  %6 = select fast i1 %5, double %2, double %3
  ret double %6
}

; Function Attrs: norecurse nounwind readnone
define double @select_cc_double_double(double %0, double %1, double %2, double %3) {
; CHECK-LABEL: select_cc_double_double:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    fcmp.d %s0, %s0, %s1
; CHECK-NEXT:    cmov.d.eq %s3, %s2, %s0
; CHECK-NEXT:    or %s0, 0, %s3
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = fcmp fast oeq double %0, %1
  %6 = select fast i1 %5, double %2, double %3
  ret double %6
}

; Function Attrs: norecurse nounwind readnone
define double @select_cc_quad_double(fp128 %0, fp128 %1, double %2, double %3) {
; CHECK-LABEL: select_cc_quad_double:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    fcmp.q %s0, %s0, %s2
; CHECK-NEXT:    cmov.d.eq %s5, %s4, %s0
; CHECK-NEXT:    or %s0, 0, %s5
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = fcmp fast oeq fp128 %0, %1
  %6 = select fast i1 %5, double %2, double %3
  ret double %6
}

; Function Attrs: norecurse nounwind readnone
define fp128 @select_cc_i1_quad(i1 zeroext %0, i1 zeroext %1, fp128 %2, fp128 %3) {
; CHECK-LABEL: select_cc_i1_quad:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    xor %s0, %s0, %s1
; CHECK-NEXT:    adds.w.sx %s0, %s0, (0)1
; CHECK-NEXT:    cmov.w.ne %s2, %s4, %s0
; CHECK-NEXT:    cmov.w.ne %s3, %s5, %s0
; CHECK-NEXT:    or %s0, 0, %s2
; CHECK-NEXT:    or %s1, 0, %s3
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = xor i1 %0, %1
  %6 = select fast i1 %5, fp128 %3, fp128 %2
  ret fp128 %6
}

; Function Attrs: norecurse nounwind readnone
define fp128 @select_cc_i8_quad(i8 signext %0, i8 signext %1, fp128 %2, fp128 %3) {
; CHECK-LABEL: select_cc_i8_quad:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    cmps.w.sx %s0, %s0, %s1
; CHECK-NEXT:    cmov.w.eq %s4, %s2, %s0
; CHECK-NEXT:    cmov.w.eq %s5, %s3, %s0
; CHECK-NEXT:    or %s0, 0, %s4
; CHECK-NEXT:    or %s1, 0, %s5
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = icmp eq i8 %0, %1
  %6 = select fast i1 %5, fp128 %2, fp128 %3
  ret fp128 %6
}

; Function Attrs: norecurse nounwind readnone
define fp128 @select_cc_u8_quad(i8 zeroext %0, i8 zeroext %1, fp128 %2, fp128 %3) {
; CHECK-LABEL: select_cc_u8_quad:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    cmps.w.sx %s0, %s0, %s1
; CHECK-NEXT:    cmov.w.eq %s4, %s2, %s0
; CHECK-NEXT:    cmov.w.eq %s5, %s3, %s0
; CHECK-NEXT:    or %s0, 0, %s4
; CHECK-NEXT:    or %s1, 0, %s5
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = icmp eq i8 %0, %1
  %6 = select fast i1 %5, fp128 %2, fp128 %3
  ret fp128 %6
}

; Function Attrs: norecurse nounwind readnone
define fp128 @select_cc_i16_quad(i16 signext %0, i16 signext %1, fp128 %2, fp128 %3) {
; CHECK-LABEL: select_cc_i16_quad:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    cmps.w.sx %s0, %s0, %s1
; CHECK-NEXT:    cmov.w.eq %s4, %s2, %s0
; CHECK-NEXT:    cmov.w.eq %s5, %s3, %s0
; CHECK-NEXT:    or %s0, 0, %s4
; CHECK-NEXT:    or %s1, 0, %s5
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = icmp eq i16 %0, %1
  %6 = select fast i1 %5, fp128 %2, fp128 %3
  ret fp128 %6
}

; Function Attrs: norecurse nounwind readnone
define fp128 @select_cc_u16_quad(i16 zeroext %0, i16 zeroext %1, fp128 %2, fp128 %3) {
; CHECK-LABEL: select_cc_u16_quad:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    cmps.w.sx %s0, %s0, %s1
; CHECK-NEXT:    cmov.w.eq %s4, %s2, %s0
; CHECK-NEXT:    cmov.w.eq %s5, %s3, %s0
; CHECK-NEXT:    or %s0, 0, %s4
; CHECK-NEXT:    or %s1, 0, %s5
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = icmp eq i16 %0, %1
  %6 = select fast i1 %5, fp128 %2, fp128 %3
  ret fp128 %6
}

; Function Attrs: norecurse nounwind readnone
define fp128 @select_cc_i32_quad(i32 signext %0, i32 signext %1, fp128 %2, fp128 %3) {
; CHECK-LABEL: select_cc_i32_quad:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    cmps.w.sx %s0, %s0, %s1
; CHECK-NEXT:    cmov.w.eq %s4, %s2, %s0
; CHECK-NEXT:    cmov.w.eq %s5, %s3, %s0
; CHECK-NEXT:    or %s0, 0, %s4
; CHECK-NEXT:    or %s1, 0, %s5
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = icmp eq i32 %0, %1
  %6 = select fast i1 %5, fp128 %2, fp128 %3
  ret fp128 %6
}

; Function Attrs: norecurse nounwind readnone
define fp128 @select_cc_u32_quad(i32 zeroext %0, i32 zeroext %1, fp128 %2, fp128 %3) {
; CHECK-LABEL: select_cc_u32_quad:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    cmps.w.sx %s0, %s0, %s1
; CHECK-NEXT:    cmov.w.eq %s4, %s2, %s0
; CHECK-NEXT:    cmov.w.eq %s5, %s3, %s0
; CHECK-NEXT:    or %s0, 0, %s4
; CHECK-NEXT:    or %s1, 0, %s5
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = icmp eq i32 %0, %1
  %6 = select fast i1 %5, fp128 %2, fp128 %3
  ret fp128 %6
}

; Function Attrs: norecurse nounwind readnone
define fp128 @select_cc_i64_quad(i64 %0, i64 %1, fp128 %2, fp128 %3) {
; CHECK-LABEL: select_cc_i64_quad:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    cmps.l %s0, %s0, %s1
; CHECK-NEXT:    cmov.l.eq %s4, %s2, %s0
; CHECK-NEXT:    cmov.l.eq %s5, %s3, %s0
; CHECK-NEXT:    or %s0, 0, %s4
; CHECK-NEXT:    or %s1, 0, %s5
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = icmp eq i64 %0, %1
  %6 = select fast i1 %5, fp128 %2, fp128 %3
  ret fp128 %6
}

; Function Attrs: norecurse nounwind readnone
define fp128 @select_cc_u64_quad(i64 %0, i64 %1, fp128 %2, fp128 %3) {
; CHECK-LABEL: select_cc_u64_quad:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    cmps.l %s0, %s0, %s1
; CHECK-NEXT:    cmov.l.eq %s4, %s2, %s0
; CHECK-NEXT:    cmov.l.eq %s5, %s3, %s0
; CHECK-NEXT:    or %s0, 0, %s4
; CHECK-NEXT:    or %s1, 0, %s5
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = icmp eq i64 %0, %1
  %6 = select fast i1 %5, fp128 %2, fp128 %3
  ret fp128 %6
}

; Function Attrs: norecurse nounwind readnone
define fp128 @select_cc_i128_quad(i128 %0, i128 %1, fp128 %2, fp128 %3) {
; CHECK-LABEL: select_cc_i128_quad:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    xor %s1, %s1, %s3
; CHECK-NEXT:    xor %s0, %s0, %s2
; CHECK-NEXT:    or %s0, %s0, %s1
; CHECK-NEXT:    cmps.l %s0, %s0, (0)1
; CHECK-NEXT:    cmov.l.eq %s6, %s4, %s0
; CHECK-NEXT:    cmov.l.eq %s7, %s5, %s0
; CHECK-NEXT:    or %s0, 0, %s6
; CHECK-NEXT:    or %s1, 0, %s7
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = icmp eq i128 %0, %1
  %6 = select fast i1 %5, fp128 %2, fp128 %3
  ret fp128 %6
}

; Function Attrs: norecurse nounwind readnone
define fp128 @select_cc_u128_quad(i128 %0, i128 %1, fp128 %2, fp128 %3) {
; CHECK-LABEL: select_cc_u128_quad:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    xor %s1, %s1, %s3
; CHECK-NEXT:    xor %s0, %s0, %s2
; CHECK-NEXT:    or %s0, %s0, %s1
; CHECK-NEXT:    cmps.l %s0, %s0, (0)1
; CHECK-NEXT:    cmov.l.eq %s6, %s4, %s0
; CHECK-NEXT:    cmov.l.eq %s7, %s5, %s0
; CHECK-NEXT:    or %s0, 0, %s6
; CHECK-NEXT:    or %s1, 0, %s7
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = icmp eq i128 %0, %1
  %6 = select fast i1 %5, fp128 %2, fp128 %3
  ret fp128 %6
}

; Function Attrs: norecurse nounwind readnone
define fp128 @select_cc_float_quad(float %0, float %1, fp128 %2, fp128 %3) {
; CHECK-LABEL: select_cc_float_quad:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    fcmp.s %s0, %s0, %s1
; CHECK-NEXT:    cmov.s.eq %s4, %s2, %s0
; CHECK-NEXT:    cmov.s.eq %s5, %s3, %s0
; CHECK-NEXT:    or %s0, 0, %s4
; CHECK-NEXT:    or %s1, 0, %s5
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = fcmp fast oeq float %0, %1
  %6 = select fast i1 %5, fp128 %2, fp128 %3
  ret fp128 %6
}

; Function Attrs: norecurse nounwind readnone
define fp128 @select_cc_double_quad(double %0, double %1, fp128 %2, fp128 %3) {
; CHECK-LABEL: select_cc_double_quad:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    fcmp.d %s0, %s0, %s1
; CHECK-NEXT:    cmov.d.eq %s4, %s2, %s0
; CHECK-NEXT:    cmov.d.eq %s5, %s3, %s0
; CHECK-NEXT:    or %s0, 0, %s4
; CHECK-NEXT:    or %s1, 0, %s5
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = fcmp fast oeq double %0, %1
  %6 = select fast i1 %5, fp128 %2, fp128 %3
  ret fp128 %6
}

; Function Attrs: norecurse nounwind readnone
define fp128 @select_cc_quad_quad(fp128 %0, fp128 %1, fp128 %2, fp128 %3) {
; CHECK-LABEL: select_cc_quad_quad:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    fcmp.q %s0, %s0, %s2
; CHECK-NEXT:    cmov.d.eq %s6, %s4, %s0
; CHECK-NEXT:    cmov.d.eq %s7, %s5, %s0
; CHECK-NEXT:    or %s0, 0, %s6
; CHECK-NEXT:    or %s1, 0, %s7
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = fcmp fast oeq fp128 %0, %1
  %6 = select fast i1 %5, fp128 %2, fp128 %3
  ret fp128 %6
}
