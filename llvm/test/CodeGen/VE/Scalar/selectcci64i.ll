; RUN: llc < %s -mtriple=ve-unknown-unknown | FileCheck %s

define i64 @selectcceq(i64, i64, i64, i64) {
; CHECK-LABEL: selectcceq:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cmpu.l %s0, 12, %s0
; CHECK-NEXT:    cmov.l.eq %s3, %s2, %s0
; CHECK-NEXT:    or %s0, 0, %s3
; CHECK-NEXT:    b.l.t (, %s10)
  %5 = icmp eq i64 %0, 12
  %6 = select i1 %5, i64 %2, i64 %3
  ret i64 %6
}

define i64 @selectccne(i64, i64, i64, i64) {
; CHECK-LABEL: selectccne:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cmpu.l %s0, 12, %s0
; CHECK-NEXT:    cmov.l.ne %s3, %s2, %s0
; CHECK-NEXT:    or %s0, 0, %s3
; CHECK-NEXT:    b.l.t (, %s10)
  %5 = icmp ne i64 %0, 12
  %6 = select i1 %5, i64 %2, i64 %3
  ret i64 %6
}

define i64 @selectccsgt(i64, i64, i64, i64) {
; CHECK-LABEL: selectccsgt:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cmps.l %s0, 12, %s0
; CHECK-NEXT:    cmov.l.lt %s3, %s2, %s0
; CHECK-NEXT:    or %s0, 0, %s3
; CHECK-NEXT:    b.l.t (, %s10)
  %5 = icmp sgt i64 %0, 12
  %6 = select i1 %5, i64 %2, i64 %3
  ret i64 %6
}

define i64 @selectccsge(i64, i64, i64, i64) {
; CHECK-LABEL: selectccsge:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cmps.l %s0, 11, %s0
; CHECK-NEXT:    cmov.l.lt %s3, %s2, %s0
; CHECK-NEXT:    or %s0, 0, %s3
; CHECK-NEXT:    b.l.t (, %s10)
  %5 = icmp sge i64 %0, 12
  %6 = select i1 %5, i64 %2, i64 %3
  ret i64 %6
}

define i64 @selectccslt(i64, i64, i64, i64) {
; CHECK-LABEL: selectccslt:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cmps.l %s0, 12, %s0
; CHECK-NEXT:    cmov.l.gt %s3, %s2, %s0
; CHECK-NEXT:    or %s0, 0, %s3
; CHECK-NEXT:    b.l.t (, %s10)
  %5 = icmp slt i64 %0, 12
  %6 = select i1 %5, i64 %2, i64 %3
  ret i64 %6
}

define i64 @selectccsle(i64, i64, i64, i64) {
; CHECK-LABEL: selectccsle:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cmps.l %s0, 13, %s0
; CHECK-NEXT:    cmov.l.gt %s3, %s2, %s0
; CHECK-NEXT:    or %s0, 0, %s3
; CHECK-NEXT:    b.l.t (, %s10)
  %5 = icmp sle i64 %0, 12
  %6 = select i1 %5, i64 %2, i64 %3
  ret i64 %6
}

define i64 @selectccugt(i64, i64, i64, i64) {
; CHECK-LABEL: selectccugt:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cmpu.l %s0, 12, %s0
; CHECK-NEXT:    cmov.l.lt %s3, %s2, %s0
; CHECK-NEXT:    or %s0, 0, %s3
; CHECK-NEXT:    b.l.t (, %s10)
  %5 = icmp ugt i64 %0, 12
  %6 = select i1 %5, i64 %2, i64 %3
  ret i64 %6
}

define i64 @selectccuge(i64, i64, i64, i64) {
; CHECK-LABEL: selectccuge:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cmpu.l %s0, 11, %s0
; CHECK-NEXT:    cmov.l.lt %s3, %s2, %s0
; CHECK-NEXT:    or %s0, 0, %s3
; CHECK-NEXT:    b.l.t (, %s10)
  %5 = icmp uge i64 %0, 12
  %6 = select i1 %5, i64 %2, i64 %3
  ret i64 %6
}

define i64 @selectccult(i64, i64, i64, i64) {
; CHECK-LABEL: selectccult:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cmpu.l %s0, 12, %s0
; CHECK-NEXT:    cmov.l.gt %s3, %s2, %s0
; CHECK-NEXT:    or %s0, 0, %s3
; CHECK-NEXT:    b.l.t (, %s10)
  %5 = icmp ult i64 %0, 12
  %6 = select i1 %5, i64 %2, i64 %3
  ret i64 %6
}

define i64 @selectccule(i64, i64, i64, i64) {
; CHECK-LABEL: selectccule:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cmpu.l %s0, 13, %s0
; CHECK-NEXT:    cmov.l.gt %s3, %s2, %s0
; CHECK-NEXT:    or %s0, 0, %s3
; CHECK-NEXT:    b.l.t (, %s10)
  %5 = icmp ule i64 %0, 12
  %6 = select i1 %5, i64 %2, i64 %3
  ret i64 %6
}

define i64 @selectccugt2(i64, i64, i64, i64) {
; CHECK-LABEL: selectccugt2:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cmpu.l %s0, 12, %s0
; CHECK-NEXT:    cmov.l.lt %s3, %s2, %s0
; CHECK-NEXT:    or %s0, 0, %s3
; CHECK-NEXT:    b.l.t (, %s10)
  %5 = icmp ugt i64 %0, 12
  %6 = select i1 %5, i64 %2, i64 %3
  ret i64 %6
}

define i64 @selectccuge2(i64, i64, i64, i64) {
; CHECK-LABEL: selectccuge2:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cmpu.l %s0, 11, %s0
; CHECK-NEXT:    cmov.l.lt %s3, %s2, %s0
; CHECK-NEXT:    or %s0, 0, %s3
; CHECK-NEXT:    b.l.t (, %s10)
  %5 = icmp uge i64 %0, 12
  %6 = select i1 %5, i64 %2, i64 %3
  ret i64 %6
}

define i64 @selectccult2(i64, i64, i64, i64) {
; CHECK-LABEL: selectccult2:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cmpu.l %s0, 12, %s0
; CHECK-NEXT:    cmov.l.gt %s3, %s2, %s0
; CHECK-NEXT:    or %s0, 0, %s3
; CHECK-NEXT:    b.l.t (, %s10)
  %5 = icmp ult i64 %0, 12
  %6 = select i1 %5, i64 %2, i64 %3
  ret i64 %6
}

define i64 @selectccule2(i64, i64, i64, i64) {
; CHECK-LABEL: selectccule2:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cmpu.l %s0, 13, %s0
; CHECK-NEXT:    cmov.l.gt %s3, %s2, %s0
; CHECK-NEXT:    or %s0, 0, %s3
; CHECK-NEXT:    b.l.t (, %s10)
  %5 = icmp ule i64 %0, 12
  %6 = select i1 %5, i64 %2, i64 %3
  ret i64 %6
}
