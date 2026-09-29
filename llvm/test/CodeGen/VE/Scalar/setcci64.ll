; RUN: llc < %s -mtriple=ve-unknown-unknown | FileCheck %s

define zeroext i1 @setcceq(i64, i64) {
; CHECK-LABEL: setcceq:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cmps.l %s0, %s0, %s1
; CHECK-NEXT:    or %s1, 0, (0)1
; CHECK-NEXT:    cmov.l.eq %s1, (63)0, %s0
; CHECK-NEXT:    adds.w.zx %s0, %s1, (0)1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = icmp eq i64 %0, %1
  ret i1 %3
}

define zeroext i1 @setccne(i64, i64) {
; CHECK-LABEL: setccne:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cmps.l %s0, %s0, %s1
; CHECK-NEXT:    or %s1, 0, (0)1
; CHECK-NEXT:    cmov.l.ne %s1, (63)0, %s0
; CHECK-NEXT:    adds.w.zx %s0, %s1, (0)1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = icmp ne i64 %0, %1
  ret i1 %3
}

define zeroext i1 @setccugt(i64, i64) {
; CHECK-LABEL: setccugt:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cmpu.l %s0, %s0, %s1
; CHECK-NEXT:    or %s1, 0, (0)1
; CHECK-NEXT:    cmov.l.gt %s1, (63)0, %s0
; CHECK-NEXT:    adds.w.zx %s0, %s1, (0)1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = icmp ugt i64 %0, %1
  ret i1 %3
}

define zeroext i1 @setccuge(i64, i64) {
; CHECK-LABEL: setccuge:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cmpu.l %s0, %s0, %s1
; CHECK-NEXT:    or %s1, 0, (0)1
; CHECK-NEXT:    cmov.l.ge %s1, (63)0, %s0
; CHECK-NEXT:    adds.w.zx %s0, %s1, (0)1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = icmp uge i64 %0, %1
  ret i1 %3
}

define zeroext i1 @setccult(i64, i64) {
; CHECK-LABEL: setccult:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cmpu.l %s0, %s0, %s1
; CHECK-NEXT:    or %s1, 0, (0)1
; CHECK-NEXT:    cmov.l.lt %s1, (63)0, %s0
; CHECK-NEXT:    adds.w.zx %s0, %s1, (0)1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = icmp ult i64 %0, %1
  ret i1 %3
}

define zeroext i1 @setccule(i64, i64) {
; CHECK-LABEL: setccule:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cmpu.l %s0, %s0, %s1
; CHECK-NEXT:    or %s1, 0, (0)1
; CHECK-NEXT:    cmov.l.le %s1, (63)0, %s0
; CHECK-NEXT:    adds.w.zx %s0, %s1, (0)1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = icmp ule i64 %0, %1
  ret i1 %3
}

define zeroext i1 @setccsgt(i64, i64) {
; CHECK-LABEL: setccsgt:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cmps.l %s0, %s0, %s1
; CHECK-NEXT:    or %s1, 0, (0)1
; CHECK-NEXT:    cmov.l.gt %s1, (63)0, %s0
; CHECK-NEXT:    adds.w.zx %s0, %s1, (0)1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = icmp sgt i64 %0, %1
  ret i1 %3
}

define zeroext i1 @setccsge(i64, i64) {
; CHECK-LABEL: setccsge:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cmps.l %s0, %s0, %s1
; CHECK-NEXT:    or %s1, 0, (0)1
; CHECK-NEXT:    cmov.l.ge %s1, (63)0, %s0
; CHECK-NEXT:    adds.w.zx %s0, %s1, (0)1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = icmp sge i64 %0, %1
  ret i1 %3
}

define zeroext i1 @setccslt(i64, i64) {
; CHECK-LABEL: setccslt:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cmps.l %s0, %s0, %s1
; CHECK-NEXT:    or %s1, 0, (0)1
; CHECK-NEXT:    cmov.l.lt %s1, (63)0, %s0
; CHECK-NEXT:    adds.w.zx %s0, %s1, (0)1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = icmp slt i64 %0, %1
  ret i1 %3
}

define zeroext i1 @setccsle(i64, i64) {
; CHECK-LABEL: setccsle:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cmps.l %s0, %s0, %s1
; CHECK-NEXT:    or %s1, 0, (0)1
; CHECK-NEXT:    cmov.l.le %s1, (63)0, %s0
; CHECK-NEXT:    adds.w.zx %s0, %s1, (0)1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = icmp sle i64 %0, %1
  ret i1 %3
}
