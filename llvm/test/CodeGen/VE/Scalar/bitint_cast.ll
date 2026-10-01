; RUN: llc < %s -mtriple=ve-unknown-unknown | FileCheck %s

define signext i32 @trunc_i65_to_i32(i65 %0) {
; CHECK-LABEL: trunc_i65_to_i32:
; CHECK:       # %bb.0:
; CHECK-NEXT:    adds.w.sx %s0, %s0, (0)1
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = trunc i65 %0 to i32
  ret i32 %2
}

define i65 @sext_i32_to_i65(i32 signext %0) {
; CHECK-LABEL: sext_i32_to_i65:
; CHECK:       # %bb.0:
; CHECK-NEXT:    sra.l %s1, %s0, 63
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = sext i32 %0 to i65
  ret i65 %2
}

define i65 @zext_i32_to_i65(i32 zeroext %0) {
; CHECK-LABEL: zext_i32_to_i65:
; CHECK:       # %bb.0:
; CHECK-NEXT:    or %s1, 0, (0)1
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = zext i32 %0 to i65
  ret i65 %2
}

define signext i17 @trunc_i64_to_i17(i64 %0) {
; CHECK-LABEL: trunc_i64_to_i17:
; CHECK:       # %bb.0:
; CHECK-NEXT:    sll %s0, %s0, 47
; CHECK-NEXT:    sra.l %s0, %s0, 47
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = trunc i64 %0 to i17
  ret i17 %2
}

define i64 @sext_i17_to_i64(i17 signext %0) {
; CHECK-LABEL: sext_i17_to_i64:
; CHECK:       # %bb.0:
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = sext i17 %0 to i64
  ret i64 %2
}

define i64 @zext_i17_to_i64(i17 zeroext %0) {
; CHECK-LABEL: zext_i17_to_i64:
; CHECK:       # %bb.0:
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = zext i17 %0 to i64
  ret i64 %2
}

define i128 @sext_i77_to_i128(i77 %0) {
; CHECK-LABEL: sext_i77_to_i128:
; CHECK:       # %bb.0:
; CHECK-NEXT:    sll %s1, %s1, 51
; CHECK-NEXT:    sra.l %s1, %s1, 51
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = sext i77 %0 to i128
  ret i128 %2
}

define i128 @zext_i77_to_i128(i77 %0) {
; CHECK-LABEL: zext_i77_to_i128:
; CHECK:       # %bb.0:
; CHECK-NEXT:    and %s1, %s1, (51)0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = zext i77 %0 to i128
  ret i128 %2
}

define signext i32 @trunc_i128_to_i32(i128 %0) {
; CHECK-LABEL: trunc_i128_to_i32:
; CHECK:       # %bb.0:
; CHECK-NEXT:    adds.w.sx %s0, %s0, (0)1
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = trunc i128 %0 to i32
  ret i32 %2
}

define i128 @sext_i64_to_i128(i64 %0) {
; CHECK-LABEL: sext_i64_to_i128:
; CHECK:       # %bb.0:
; CHECK-NEXT:    sra.l %s1, %s0, 63
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = sext i64 %0 to i128
  ret i128 %2
}

define i128 @zext_i64_to_i128(i64 %0) {
; CHECK-LABEL: zext_i64_to_i128:
; CHECK:       # %bb.0:
; CHECK-NEXT:    or %s1, 0, (0)1
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = zext i64 %0 to i128
  ret i128 %2
}

define signext i32 @trunc_i131_to_i32(i131 %0) {
; CHECK-LABEL: trunc_i131_to_i32:
; CHECK:       # %bb.0:
; CHECK-NEXT:    adds.w.sx %s0, %s0, (0)1
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = trunc i131 %0 to i32
  ret i32 %2
}

define i131 @sext_i64_to_i131(i64 %0) {
; CHECK-LABEL: sext_i64_to_i131:
; CHECK:       # %bb.0:
; CHECK-NEXT:    sra.l %s1, %s0, 63
; CHECK-NEXT:    or %s2, 0, %s1
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = sext i64 %0 to i131
  ret i131 %2
}

define i131 @zext_i64_to_i131(i64 %0) {
; CHECK-LABEL: zext_i64_to_i131:
; CHECK:       # %bb.0:
; CHECK-NEXT:    or %s1, 0, (0)1
; CHECK-NEXT:    or %s2, 0, (0)1
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = zext i64 %0 to i131
  ret i131 %2
}

; i131 float-to-int and int-to-float casts are expanded inline, producing
; very large code. We verify fptoui_double_to_i131 with full CHECK lines
; and the rest only check they compile without crashing.

define i131 @fptoui_double_to_i131(double %a) {
; CHECK-LABEL: fptoui_double_to_i131:
; CHECK:       # %bb.0:
; CHECK-NEXT:    adds.l %s11, -64, %s11
; CHECK-NEXT:    brge.l %s11, %s8, .LBB{{[0-9]+}}_8
; CHECK:       # %bb.7:
; CHECK:         monc
; CHECK-NEXT:    or %s0, 0, %s62
; CHECK-NEXT:  .LBB{{[0-9]+}}_8:
; CHECK-NEXT:    srl %s1, %s0, 52
; CHECK-NEXT:    and %s1, %s1, (53)0
; CHECK-NEXT:    lea %s2, 1023
; CHECK-NEXT:    cmpu.l %s2, %s2, %s1
; CHECK-NEXT:    brge.l 0, %s2, .LBB{{[0-9]+}}_3
; CHECK-NEXT:  # %bb.1:
; CHECK-NEXT:    or %s0, 0, (0)1
; CHECK-NEXT:    br.l.t .LBB{{[0-9]+}}_2
; CHECK-NEXT:  .LBB{{[0-9]+}}_3:
; CHECK-NEXT:    and %s0, %s0, (12)0
; CHECK-NEXT:    lea.sl %s2, 1048576
; CHECK-NEXT:    lea %s3, 1074
; CHECK-NEXT:    cmpu.l %s3, %s3, %s1
; CHECK-NEXT:    or %s0, %s0, %s2
; CHECK-NEXT:    brgt.l 0, %s3, .LBB{{[0-9]+}}_5
; CHECK-NEXT:  # %bb.4:
; CHECK-NEXT:    lea %s2, 1075
; CHECK-NEXT:    subs.w.sx %s1, %s2, %s1
; CHECK-NEXT:    srl %s0, %s0, %s1
; CHECK-NEXT:  .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    or %s1, 0, (0)1
; CHECK-NEXT:    or %s2, 0, (0)1
; CHECK-NEXT:    lea %s11, 64(, %s11)
; CHECK-NEXT:    b.l.t (, %s10)
; CHECK-NEXT:  .LBB{{[0-9]+}}_5:
; CHECK-NEXT:    or %s2, 0, (0)1
; CHECK-NEXT:    st %s2, 56(, %s11)
; CHECK-NEXT:    st %s2, 48(, %s11)
; CHECK-NEXT:    st %s2, 40(, %s11)
; CHECK-NEXT:    st %s0, 32(, %s11)
; CHECK-NEXT:    st %s2, 24(, %s11)
; CHECK-NEXT:    st %s2, 16(, %s11)
; CHECK-NEXT:    st %s2, 8(, %s11)
; CHECK-NEXT:    st %s2, (, %s11)
; CHECK-NEXT:    lea %s0, -1075
; CHECK-NEXT:    adds.w.sx %s0, %s1, %s0
; CHECK-NEXT:    and %s1, %s0, (32)0
; CHECK-NEXT:    srl %s1, %s1, 3
; CHECK-NEXT:    and %s1, 24, %s1
; CHECK-NEXT:    subs.w.sx %s1, 0, %s1
; CHECK-NEXT:    adds.w.sx %s1, %s1, (0)1
; CHECK-NEXT:    lea %s2, (, %s11)
; CHECK-NEXT:    lea %s2, 32(, %s2)
; CHECK-NEXT:    ld %s3, 8(%s1, %s2)
; CHECK-NEXT:    and %s0, 63, %s0
; CHECK-NEXT:    ld %s4, 32(%s1, %s11)
; CHECK-NEXT:    sll %s5, %s3, %s0
; CHECK-NEXT:    xor %s6, 63, %s0
; CHECK-NEXT:    ld %s2, 16(%s1, %s2)
; CHECK-NEXT:    srl %s1, %s4, 1
; CHECK-NEXT:    srl %s1, %s1, %s6
; CHECK-NEXT:    or %s1, %s5, %s1
; CHECK-NEXT:    sll %s2, %s2, %s0
; CHECK-NEXT:    srl %s3, %s3, 1
; CHECK-NEXT:    srl %s3, %s3, %s6
; CHECK-NEXT:    or %s2, %s2, %s3
; CHECK-NEXT:    and %s2, 7, %s2
; CHECK-NEXT:    sll %s0, %s4, %s0
; CHECK-NEXT:    lea %s11, 64(, %s11)
; CHECK-NEXT:    b.l.t (, %s10)
  %conv = fptoui double %a to i131
  ret i131 %conv
}

define float @sitofp_i131_to_float(i131 %a) {
; CHECK-LABEL: sitofp_i131_to_float:
; CHECK:         b.l.t (, %s10)
  %conv = sitofp i131 %a to float
  ret float %conv
}

define double @sitofp_i131_to_double(i131 %a) {
; CHECK-LABEL: sitofp_i131_to_double:
; CHECK:         b.l.t (, %s10)
  %conv = sitofp i131 %a to double
  ret double %conv
}

define float @uitofp_i131_to_float(i131 %a) {
; CHECK-LABEL: uitofp_i131_to_float:
; CHECK:         b.l.t (, %s10)
  %conv = uitofp i131 %a to float
  ret float %conv
}

define double @uitofp_i131_to_double(i131 %a) {
; CHECK-LABEL: uitofp_i131_to_double:
; CHECK:         b.l.t (, %s10)
  %conv = uitofp i131 %a to double
  ret double %conv
}

define i131 @fptosi_float_to_i131(float %a) {
; CHECK-LABEL: fptosi_float_to_i131:
; CHECK:         b.l.t (, %s10)
  %conv = fptosi float %a to i131
  ret i131 %conv
}

define i131 @fptosi_double_to_i131(double %a) {
; CHECK-LABEL: fptosi_double_to_i131:
; CHECK:         b.l.t (, %s10)
  %conv = fptosi double %a to i131
  ret i131 %conv
}

define i131 @fptoui_float_to_i131(float %a) {
; CHECK-LABEL: fptoui_float_to_i131:
; CHECK:         b.l.t (, %s10)
  %conv = fptoui float %a to i131
  ret i131 %conv
}
