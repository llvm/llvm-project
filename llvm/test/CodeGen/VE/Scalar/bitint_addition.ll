; RUN: llc < %s -mtriple=ve-unknown-unknown | FileCheck %s

define zeroext i17 @add_i17_zext(i17 zeroext %0, i17 zeroext %1) {
; CHECK-LABEL: add_i17_zext:
; CHECK:       # %bb.0:
; CHECK-NEXT:    adds.w.sx %s0, %s1, %s0
; CHECK-NEXT:    and %s0, %s0, (47)0
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = add i17 %1, %0
  ret i17 %3
}

define signext i17 @add_i17_sext(i17 signext %0, i17 signext %1) {
; CHECK-LABEL: add_i17_sext:
; CHECK:       # %bb.0:
; CHECK-NEXT:    adds.w.sx %s0, %s1, %s0
; CHECK-NEXT:    sll %s0, %s0, 47
; CHECK-NEXT:    sra.l %s0, %s0, 47
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = add nsw i17 %1, %0
  ret i17 %3
}

define i65 @add_i65(i65 %0, i65 %1) {
; CHECK-LABEL: add_i65:
; CHECK:       # %bb.0:
; CHECK-NEXT:    adds.l %s1, %s3, %s1
; CHECK-NEXT:    adds.l %s0, %s2, %s0
; CHECK-NEXT:    cmpu.l %s2, %s0, %s2
; CHECK-NEXT:    or %s3, 0, (0)1
; CHECK-NEXT:    cmov.l.lt %s3, (63)0, %s2
; CHECK-NEXT:    adds.w.zx %s2, %s3, (0)1
; CHECK-NEXT:    adds.l %s1, %s1, %s2
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = add i65 %1, %0
  ret i65 %3
}

define i77 @add_i77(i77 %0, i77 %1) {
; CHECK-LABEL: add_i77:
; CHECK:       # %bb.0:
; CHECK-NEXT:    adds.l %s1, %s3, %s1
; CHECK-NEXT:    adds.l %s0, %s2, %s0
; CHECK-NEXT:    cmpu.l %s2, %s0, %s2
; CHECK-NEXT:    or %s3, 0, (0)1
; CHECK-NEXT:    cmov.l.lt %s3, (63)0, %s2
; CHECK-NEXT:    adds.w.zx %s2, %s3, (0)1
; CHECK-NEXT:    adds.l %s1, %s1, %s2
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = add i77 %1, %0
  ret i77 %3
}
