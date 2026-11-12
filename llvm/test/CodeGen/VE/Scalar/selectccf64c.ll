; RUN: llc < %s -mtriple=ve-unknown-unknown | FileCheck %s

define double @selectccsgti8(i8, i8, double, double) {
; CHECK-LABEL: selectccsgti8:
; CHECK:       # %bb.0:
; CHECK-NEXT:    sll %s1, %s1, 56
; CHECK-NEXT:    sra.l %s1, %s1, 56
; CHECK-NEXT:    sll %s0, %s0, 56
; CHECK-NEXT:    sra.l %s0, %s0, 56
; CHECK-NEXT:    cmps.w.zx %s0, %s0, %s1
; CHECK-NEXT:    cmov.w.gt %s3, %s2, %s0
; CHECK-NEXT:    or %s0, 0, %s3
; CHECK-NEXT:    b.l.t (, %s10)
  %5 = icmp sgt i8 %0, %1
  %6 = select i1 %5, double %2, double %3
  ret double %6
}

define double @selectccsgti16(i16, i16, double, double) {
; CHECK-LABEL: selectccsgti16:
; CHECK:       # %bb.0:
; CHECK-NEXT:    sll %s1, %s1, 48
; CHECK-NEXT:    sra.l %s1, %s1, 48
; CHECK-NEXT:    sll %s0, %s0, 48
; CHECK-NEXT:    sra.l %s0, %s0, 48
; CHECK-NEXT:    cmps.w.zx %s0, %s0, %s1
; CHECK-NEXT:    cmov.w.gt %s3, %s2, %s0
; CHECK-NEXT:    or %s0, 0, %s3
; CHECK-NEXT:    b.l.t (, %s10)
  %5 = icmp sgt i16 %0, %1
  %6 = select i1 %5, double %2, double %3
  ret double %6
}

define double @selectccsgti32(i32, i32, double, double) {
; CHECK-LABEL: selectccsgti32:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cmps.w.zx %s0, %s0, %s1
; CHECK-NEXT:    cmov.w.gt %s3, %s2, %s0
; CHECK-NEXT:    or %s0, 0, %s3
; CHECK-NEXT:    b.l.t (, %s10)
  %5 = icmp sgt i32 %0, %1
  %6 = select i1 %5, double %2, double %3
  ret double %6
}

define double @selectccsgti64(i64, i64, double, double) {
; CHECK-LABEL: selectccsgti64:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cmps.l %s0, %s0, %s1
; CHECK-NEXT:    cmov.l.gt %s3, %s2, %s0
; CHECK-NEXT:    or %s0, 0, %s3
; CHECK-NEXT:    b.l.t (, %s10)
  %5 = icmp sgt i64 %0, %1
  %6 = select i1 %5, double %2, double %3
  ret double %6
}

define double @selectccsgti128(i128, i128, double, double) {
; CHECK-LABEL: selectccsgti128:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cmpu.l %s6, %s1, %s3
; CHECK-NEXT:    cmps.l %s1, %s1, %s3
; CHECK-NEXT:    or %s3, 0, (0)1
; CHECK-NEXT:    or %s7, 0, (0)1
; CHECK-NEXT:    cmov.l.gt %s7, (63)0, %s1
; CHECK-NEXT:    cmpu.l %s0, %s0, %s2
; CHECK-NEXT:    cmov.l.gt %s3, (63)0, %s0
; CHECK-NEXT:    cmov.l.eq %s7, %s3, %s6
; CHECK-NEXT:    cmov.w.ne %s5, %s4, %s7
; CHECK-NEXT:    or %s0, 0, %s5
; CHECK-NEXT:    b.l.t (, %s10)
  %5 = icmp sgt i128 %0, %1
  %6 = select i1 %5, double %2, double %3
  ret double %6
}

define double @selectccogtf32(float, float, double, double) {
; CHECK-LABEL: selectccogtf32:
; CHECK:       # %bb.0:
; CHECK-NEXT:    fcmp.s %s0, %s0, %s1
; CHECK-NEXT:    cmov.s.gt %s3, %s2, %s0
; CHECK-NEXT:    or %s0, 0, %s3
; CHECK-NEXT:    b.l.t (, %s10)
  %5 = fcmp ogt float %0, %1
  %6 = select i1 %5, double %2, double %3
  ret double %6
}

define double @selectccogtf64(double, double, double, double) {
; CHECK-LABEL: selectccogtf64:
; CHECK:       # %bb.0:
; CHECK-NEXT:    fcmp.d %s0, %s0, %s1
; CHECK-NEXT:    cmov.d.gt %s3, %s2, %s0
; CHECK-NEXT:    or %s0, 0, %s3
; CHECK-NEXT:    b.l.t (, %s10)
  %5 = fcmp ogt double %0, %1
  %6 = select i1 %5, double %2, double %3
  ret double %6
}
