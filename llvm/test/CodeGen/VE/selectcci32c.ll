; RUN: llc < %s -mtriple=ve-unknown-unknown | FileCheck %s

define i32 @selectccsgti8(i8, i8, i32, i32) {
; CHECK-LABEL: selectccsgti8:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    sll %s1, %s1, 56
; CHECK-NEXT:    sra.l %s1, %s1, 56
; CHECK-NEXT:    sll %s0, %s0, 56
; CHECK-NEXT:    sra.l %s0, %s0, 56
; CHECK-NEXT:    cmps.w.sx %s0, %s0, %s1
; CHECK-NEXT:    cmov.w.gt %s3, %s2, %s0
; CHECK-NEXT:    or %s0, 0, %s3
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = icmp sgt i8 %0, %1
  %6 = select i1 %5, i32 %2, i32 %3
  ret i32 %6
}

define i32 @selectccsgti16(i16, i16, i32, i32) {
; CHECK-LABEL: selectccsgti16:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    sll %s1, %s1, 48
; CHECK-NEXT:    sra.l %s1, %s1, 48
; CHECK-NEXT:    sll %s0, %s0, 48
; CHECK-NEXT:    sra.l %s0, %s0, 48
; CHECK-NEXT:    cmps.w.sx %s0, %s0, %s1
; CHECK-NEXT:    cmov.w.gt %s3, %s2, %s0
; CHECK-NEXT:    or %s0, 0, %s3
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = icmp sgt i16 %0, %1
  %6 = select i1 %5, i32 %2, i32 %3
  ret i32 %6
}

define i32 @selectccsgti32(i32, i32, i32, i32) {
; CHECK-LABEL: selectccsgti32:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    cmps.w.sx %s0, %s0, %s1
; CHECK-NEXT:    cmov.w.gt %s3, %s2, %s0
; CHECK-NEXT:    or %s0, 0, %s3
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = icmp sgt i32 %0, %1
  %6 = select i1 %5, i32 %2, i32 %3
  ret i32 %6
}

define i32 @selectccsgti64(i64, i64, i32, i32) {
; CHECK-LABEL: selectccsgti64:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    cmps.l %s0, %s0, %s1
; CHECK-NEXT:    cmov.l.gt %s3, %s2, %s0
; CHECK-NEXT:    or %s0, 0, %s3
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = icmp sgt i64 %0, %1
  %6 = select i1 %5, i32 %2, i32 %3
  ret i32 %6
}

define i32 @selectccsgti128(i128, i128, i32, i32) {
; CHECK-LABEL: selectccsgti128:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    or %s6, 0, (0)1
; CHECK-NEXT:    cmps.l %s1, %s1, %s3
; CHECK-NEXT:    or %s3, 0, %s6
; CHECK-NEXT:    cmov.l.gt %s3, (63)0, %s1
; CHECK-NEXT:    cmpu.l %s0, %s0, %s2
; CHECK-NEXT:    or %s2, 0, %s6
; CHECK-NEXT:    cmov.l.gt %s2, (63)0, %s0
; CHECK-NEXT:    cmov.l.eq %s3, %s2, %s1
; CHECK-NEXT:    cmps.w.sx %s0, %s3, %s6
; CHECK-NEXT:    cmov.w.ne %s5, %s4, %s0
; CHECK-NEXT:    or %s0, 0, %s5
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = icmp sgt i128 %0, %1
  %6 = select i1 %5, i32 %2, i32 %3
  ret i32 %6
}

define i32 @selectccogtf32(float, float, i32, i32) {
; CHECK-LABEL: selectccogtf32:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    fcmp.s %s0, %s0, %s1
; CHECK-NEXT:    cmov.s.gt %s3, %s2, %s0
; CHECK-NEXT:    or %s0, 0, %s3
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = fcmp ogt float %0, %1
  %6 = select i1 %5, i32 %2, i32 %3
  ret i32 %6
}

define i32 @selectccogtf64(double, double, i32, i32) {
; CHECK-LABEL: selectccogtf64:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    fcmp.d %s0, %s0, %s1
; CHECK-NEXT:    cmov.d.gt %s3, %s2, %s0
; CHECK-NEXT:    or %s0, 0, %s3
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = fcmp ogt double %0, %1
  %6 = select i1 %5, i32 %2, i32 %3
  ret i32 %6
}

