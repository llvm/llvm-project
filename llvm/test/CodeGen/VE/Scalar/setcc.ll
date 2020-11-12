; RUN: llc < %s -mtriple=ve | FileCheck %s

;;; Test all combination of input type and output type among following types.
;;;
;;; Types:
;;;  i1/i8/u8/i16/u16/i32/u32/i64/u64/i128/u128/float/double/fp128

; Function Attrs: norecurse nounwind readnone
define zeroext i1 @setcc_i1(i1 zeroext %0, i1 zeroext %1) {
; CHECK-LABEL: setcc_i1:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    xor %s0, %s0, %s1
; CHECK-NEXT:    xor %s0, 1, %s0
; CHECK-NEXT:    or %s11, 0, %s9
  %3 = xor i1 %0, %1
  %4 = xor i1 %3, true
  ret i1 %4
}

; Function Attrs: norecurse nounwind readnone
define zeroext i1 @setcc_i8(i8 signext %0, i8 signext %1) {
; CHECK-LABEL: setcc_i8:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    cmps.w.sx %s0, %s0, %s1
; CHECK-NEXT:    or %s1, 0, (0)1
; CHECK-NEXT:    cmov.w.eq %s1, (63)0, %s0
; CHECK-NEXT:    adds.w.zx %s0, %s1, (0)1
; CHECK-NEXT:    or %s11, 0, %s9
  %3 = icmp eq i8 %0, %1
  ret i1 %3
}

; Function Attrs: norecurse nounwind readnone
define zeroext i1 @setcc_u8(i8 zeroext %0, i8 zeroext %1) {
; CHECK-LABEL: setcc_u8:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    cmps.w.sx %s0, %s0, %s1
; CHECK-NEXT:    or %s1, 0, (0)1
; CHECK-NEXT:    cmov.w.eq %s1, (63)0, %s0
; CHECK-NEXT:    adds.w.zx %s0, %s1, (0)1
; CHECK-NEXT:    or %s11, 0, %s9
  %3 = icmp eq i8 %0, %1
  ret i1 %3
}

; Function Attrs: norecurse nounwind readnone
define zeroext i1 @setcc_i16(i16 signext %0, i16 signext %1) {
; CHECK-LABEL: setcc_i16:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    cmps.w.sx %s0, %s0, %s1
; CHECK-NEXT:    or %s1, 0, (0)1
; CHECK-NEXT:    cmov.w.eq %s1, (63)0, %s0
; CHECK-NEXT:    adds.w.zx %s0, %s1, (0)1
; CHECK-NEXT:    or %s11, 0, %s9
  %3 = icmp eq i16 %0, %1
  ret i1 %3
}

; Function Attrs: norecurse nounwind readnone
define zeroext i1 @setcc_u16(i16 zeroext %0, i16 zeroext %1) {
; CHECK-LABEL: setcc_u16:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    cmps.w.sx %s0, %s0, %s1
; CHECK-NEXT:    or %s1, 0, (0)1
; CHECK-NEXT:    cmov.w.eq %s1, (63)0, %s0
; CHECK-NEXT:    adds.w.zx %s0, %s1, (0)1
; CHECK-NEXT:    or %s11, 0, %s9
  %3 = icmp eq i16 %0, %1
  ret i1 %3
}

; Function Attrs: norecurse nounwind readnone
define zeroext i1 @setcc_i32(i32 signext %0, i32 signext %1) {
; CHECK-LABEL: setcc_i32:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    cmps.w.sx %s0, %s0, %s1
; CHECK-NEXT:    or %s1, 0, (0)1
; CHECK-NEXT:    cmov.w.eq %s1, (63)0, %s0
; CHECK-NEXT:    adds.w.zx %s0, %s1, (0)1
; CHECK-NEXT:    or %s11, 0, %s9
  %3 = icmp eq i32 %0, %1
  ret i1 %3
}

; Function Attrs: norecurse nounwind readnone
define zeroext i1 @setcc_u32(i32 zeroext %0, i32 zeroext %1) {
; CHECK-LABEL: setcc_u32:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    cmps.w.sx %s0, %s0, %s1
; CHECK-NEXT:    or %s1, 0, (0)1
; CHECK-NEXT:    cmov.w.eq %s1, (63)0, %s0
; CHECK-NEXT:    adds.w.zx %s0, %s1, (0)1
; CHECK-NEXT:    or %s11, 0, %s9
  %3 = icmp eq i32 %0, %1
  ret i1 %3
}

; Function Attrs: norecurse nounwind readnone
define zeroext i1 @setcc_i64(i64 %0, i64 %1) {
; CHECK-LABEL: setcc_i64:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    cmps.l %s0, %s0, %s1
; CHECK-NEXT:    or %s1, 0, (0)1
; CHECK-NEXT:    cmov.l.eq %s1, (63)0, %s0
; CHECK-NEXT:    adds.w.zx %s0, %s1, (0)1
; CHECK-NEXT:    or %s11, 0, %s9
  %3 = icmp eq i64 %0, %1
  ret i1 %3
}

; Function Attrs: norecurse nounwind readnone
define zeroext i1 @setcc_u64(i64 %0, i64 %1) {
; CHECK-LABEL: setcc_u64:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    cmps.l %s0, %s0, %s1
; CHECK-NEXT:    or %s1, 0, (0)1
; CHECK-NEXT:    cmov.l.eq %s1, (63)0, %s0
; CHECK-NEXT:    adds.w.zx %s0, %s1, (0)1
; CHECK-NEXT:    or %s11, 0, %s9
  %3 = icmp eq i64 %0, %1
  ret i1 %3
}

; Function Attrs: norecurse nounwind readnone
define zeroext i1 @setcc_i128(i128 %0, i128 %1) {
; CHECK-LABEL: setcc_i128:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    xor %s1, %s1, %s3
; CHECK-NEXT:    xor %s0, %s0, %s2
; CHECK-NEXT:    or %s0, %s0, %s1
; CHECK-NEXT:    or %s1, 0, (0)1
; CHECK-NEXT:    cmps.l %s0, %s0, (0)1
; CHECK-NEXT:    cmov.l.eq %s1, (63)0, %s0
; CHECK-NEXT:    adds.w.zx %s0, %s1, (0)1
; CHECK-NEXT:    or %s11, 0, %s9
  %3 = icmp eq i128 %0, %1
  ret i1 %3
}

; Function Attrs: norecurse nounwind readnone
define zeroext i1 @setcc_u128(i128 %0, i128 %1) {
; CHECK-LABEL: setcc_u128:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    xor %s1, %s1, %s3
; CHECK-NEXT:    xor %s0, %s0, %s2
; CHECK-NEXT:    or %s0, %s0, %s1
; CHECK-NEXT:    or %s1, 0, (0)1
; CHECK-NEXT:    cmps.l %s0, %s0, (0)1
; CHECK-NEXT:    cmov.l.eq %s1, (63)0, %s0
; CHECK-NEXT:    adds.w.zx %s0, %s1, (0)1
; CHECK-NEXT:    or %s11, 0, %s9
  %3 = icmp eq i128 %0, %1
  ret i1 %3
}

; Function Attrs: norecurse nounwind readnone
define zeroext i1 @setcc_float(float %0, float %1) {
; CHECK-LABEL: setcc_float:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    fcmp.s %s0, %s0, %s1
; CHECK-NEXT:    or %s1, 0, (0)1
; CHECK-NEXT:    cmov.s.eq %s1, (63)0, %s0
; CHECK-NEXT:    adds.w.zx %s0, %s1, (0)1
; CHECK-NEXT:    or %s11, 0, %s9
  %3 = fcmp fast oeq float %0, %1
  ret i1 %3
}

; Function Attrs: norecurse nounwind readnone
define zeroext i1 @setcc_double(double %0, double %1) {
; CHECK-LABEL: setcc_double:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    fcmp.d %s0, %s0, %s1
; CHECK-NEXT:    or %s1, 0, (0)1
; CHECK-NEXT:    cmov.d.eq %s1, (63)0, %s0
; CHECK-NEXT:    adds.w.zx %s0, %s1, (0)1
; CHECK-NEXT:    or %s11, 0, %s9
  %3 = fcmp fast oeq double %0, %1
  ret i1 %3
}

; Function Attrs: norecurse nounwind readnone
define zeroext i1 @setcc_quad(fp128 %0, fp128 %1) {
; CHECK-LABEL: setcc_quad:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    fcmp.q %s0, %s0, %s2
; CHECK-NEXT:    or %s1, 0, (0)1
; CHECK-NEXT:    cmov.d.eq %s1, (63)0, %s0
; CHECK-NEXT:    adds.w.zx %s0, %s1, (0)1
; CHECK-NEXT:    or %s11, 0, %s9
  %3 = fcmp fast oeq fp128 %0, %1
  ret i1 %3
}
