; RUN: llc < %s -mtriple=ve-unknown-unknown | FileCheck %s

define i64 @selectcceq(i64, i64, i64, i64) {
; CHECK-LABEL: selectcceq:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    or %s1, 12, (0)1
; CHECK-NEXT:    cmps.l %s0, %s0, %s1
; CHECK-NEXT:    cmov.l.eq %s3, %s2, %s0
; CHECK-NEXT:    or %s0, 0, %s3
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = icmp eq i64 %0, 12
  %6 = select i1 %5, i64 %2, i64 %3
  ret i64 %6
}

define i64 @selectccne(i64, i64, i64, i64) {
; CHECK-LABEL: selectccne:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    or %s1, 12, (0)1
; CHECK-NEXT:    cmps.l %s0, %s0, %s1
; CHECK-NEXT:    cmov.l.ne %s3, %s2, %s0
; CHECK-NEXT:    or %s0, 0, %s3
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = icmp ne i64 %0, 12
  %6 = select i1 %5, i64 %2, i64 %3
  ret i64 %6
}

define i64 @selectccsgt(i64, i64, i64, i64) {
; CHECK-LABEL: selectccsgt:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    or %s1, 12, (0)1
; CHECK-NEXT:    cmps.l %s0, %s0, %s1
; CHECK-NEXT:    cmov.l.gt %s3, %s2, %s0
; CHECK-NEXT:    or %s0, 0, %s3
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = icmp sgt i64 %0, 12
  %6 = select i1 %5, i64 %2, i64 %3
  ret i64 %6
}

define i64 @selectccsge(i64, i64, i64, i64) {
; CHECK-LABEL: selectccsge:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    or %s1, 11, (0)1
; CHECK-NEXT:    cmps.l %s0, %s0, %s1
; CHECK-NEXT:    cmov.l.gt %s3, %s2, %s0
; CHECK-NEXT:    or %s0, 0, %s3
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = icmp sge i64 %0, 12
  %6 = select i1 %5, i64 %2, i64 %3
  ret i64 %6
}

define i64 @selectccslt(i64, i64, i64, i64) {
; CHECK-LABEL: selectccslt:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    or %s1, 12, (0)1
; CHECK-NEXT:    cmps.l %s0, %s0, %s1
; CHECK-NEXT:    cmov.l.lt %s3, %s2, %s0
; CHECK-NEXT:    or %s0, 0, %s3
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = icmp slt i64 %0, 12
  %6 = select i1 %5, i64 %2, i64 %3
  ret i64 %6
}

define i64 @selectccsle(i64, i64, i64, i64) {
; CHECK-LABEL: selectccsle:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    or %s1, 13, (0)1
; CHECK-NEXT:    cmps.l %s0, %s0, %s1
; CHECK-NEXT:    cmov.l.lt %s3, %s2, %s0
; CHECK-NEXT:    or %s0, 0, %s3
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = icmp sle i64 %0, 12
  %6 = select i1 %5, i64 %2, i64 %3
  ret i64 %6
}

define i64 @selectccugt(i64, i64, i64, i64) {
; CHECK-LABEL: selectccugt:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    or %s1, 12, (0)1
; CHECK-NEXT:    cmpu.l %s0, %s0, %s1
; CHECK-NEXT:    cmov.l.gt %s3, %s2, %s0
; CHECK-NEXT:    or %s0, 0, %s3
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = icmp ugt i64 %0, 12
  %6 = select i1 %5, i64 %2, i64 %3
  ret i64 %6
}

define i64 @selectccuge(i64, i64, i64, i64) {
; CHECK-LABEL: selectccuge:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    or %s1, 11, (0)1
; CHECK-NEXT:    cmpu.l %s0, %s0, %s1
; CHECK-NEXT:    cmov.l.gt %s3, %s2, %s0
; CHECK-NEXT:    or %s0, 0, %s3
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = icmp uge i64 %0, 12
  %6 = select i1 %5, i64 %2, i64 %3
  ret i64 %6
}

define i64 @selectccult(i64, i64, i64, i64) {
; CHECK-LABEL: selectccult:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    or %s1, 12, (0)1
; CHECK-NEXT:    cmpu.l %s0, %s0, %s1
; CHECK-NEXT:    cmov.l.lt %s3, %s2, %s0
; CHECK-NEXT:    or %s0, 0, %s3
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = icmp ult i64 %0, 12
  %6 = select i1 %5, i64 %2, i64 %3
  ret i64 %6
}

define i64 @selectccule(i64, i64, i64, i64) {
; CHECK-LABEL: selectccule:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    or %s1, 13, (0)1
; CHECK-NEXT:    cmpu.l %s0, %s0, %s1
; CHECK-NEXT:    cmov.l.lt %s3, %s2, %s0
; CHECK-NEXT:    or %s0, 0, %s3
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = icmp ule i64 %0, 12
  %6 = select i1 %5, i64 %2, i64 %3
  ret i64 %6
}

define i64 @selectccugt2(i64, i64, i64, i64) {
; CHECK-LABEL: selectccugt2:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    or %s1, 12, (0)1
; CHECK-NEXT:    cmpu.l %s0, %s0, %s1
; CHECK-NEXT:    cmov.l.gt %s3, %s2, %s0
; CHECK-NEXT:    or %s0, 0, %s3
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = icmp ugt i64 %0, 12
  %6 = select i1 %5, i64 %2, i64 %3
  ret i64 %6
}

define i64 @selectccuge2(i64, i64, i64, i64) {
; CHECK-LABEL: selectccuge2:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    or %s1, 11, (0)1
; CHECK-NEXT:    cmpu.l %s0, %s0, %s1
; CHECK-NEXT:    cmov.l.gt %s3, %s2, %s0
; CHECK-NEXT:    or %s0, 0, %s3
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = icmp uge i64 %0, 12
  %6 = select i1 %5, i64 %2, i64 %3
  ret i64 %6
}

define i64 @selectccult2(i64, i64, i64, i64) {
; CHECK-LABEL: selectccult2:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    or %s1, 12, (0)1
; CHECK-NEXT:    cmpu.l %s0, %s0, %s1
; CHECK-NEXT:    cmov.l.lt %s3, %s2, %s0
; CHECK-NEXT:    or %s0, 0, %s3
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = icmp ult i64 %0, 12
  %6 = select i1 %5, i64 %2, i64 %3
  ret i64 %6
}

define i64 @selectccule2(i64, i64, i64, i64) {
; CHECK-LABEL: selectccule2:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    or %s1, 13, (0)1
; CHECK-NEXT:    cmpu.l %s0, %s0, %s1
; CHECK-NEXT:    cmov.l.lt %s3, %s2, %s0
; CHECK-NEXT:    or %s0, 0, %s3
; CHECK-NEXT:    or %s11, 0, %s9
  %5 = icmp ule i64 %0, 12
  %6 = select i1 %5, i64 %2, i64 %3
  ret i64 %6
}
