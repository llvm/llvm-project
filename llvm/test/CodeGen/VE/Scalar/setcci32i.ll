; RUN: llc < %s -mtriple=ve-unknown-unknown | FileCheck %s

define zeroext i1 @setcceq(i32, i32) {
; CHECK-LABEL: setcceq:
; CHECK:       # %bb.0:
; CHECK-NEXT:    or %s1, 12, (0)1
; CHECK-NEXT:    cmps.w.sx %s0, %s0, %s1
; CHECK-NEXT:    or %s1, 0, (0)1
; CHECK-NEXT:    cmov.w.eq %s1, (63)0, %s0
; CHECK-NEXT:    adds.w.zx %s0, %s1, (0)1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = icmp eq i32 %0, 12
  ret i1 %3
}

define zeroext i1 @setccne(i32, i32) {
; CHECK-LABEL: setccne:
; CHECK:       # %bb.0:
; CHECK-NEXT:    or %s1, 12, (0)1
; CHECK-NEXT:    cmps.w.sx %s0, %s0, %s1
; CHECK-NEXT:    or %s1, 0, (0)1
; CHECK-NEXT:    cmov.w.ne %s1, (63)0, %s0
; CHECK-NEXT:    adds.w.zx %s0, %s1, (0)1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = icmp ne i32 %0, 12
  ret i1 %3
}

define zeroext i1 @setccugt(i32, i32) {
; CHECK-LABEL: setccugt:
; CHECK:       # %bb.0:
; CHECK-NEXT:    or %s1, 12, (0)1
; CHECK-NEXT:    cmpu.w %s0, %s0, %s1
; CHECK-NEXT:    or %s1, 0, (0)1
; CHECK-NEXT:    cmov.w.gt %s1, (63)0, %s0
; CHECK-NEXT:    adds.w.zx %s0, %s1, (0)1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = icmp ugt i32 %0, 12
  ret i1 %3
}

define zeroext i1 @setccuge(i32, i32) {
; CHECK-LABEL: setccuge:
; CHECK:       # %bb.0:
; CHECK-NEXT:    or %s1, 11, (0)1
; CHECK-NEXT:    cmpu.w %s0, %s0, %s1
; CHECK-NEXT:    or %s1, 0, (0)1
; CHECK-NEXT:    cmov.w.gt %s1, (63)0, %s0
; CHECK-NEXT:    adds.w.zx %s0, %s1, (0)1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = icmp uge i32 %0, 12
  ret i1 %3
}

define zeroext i1 @setccult(i32, i32) {
; CHECK-LABEL: setccult:
; CHECK:       # %bb.0:
; CHECK-NEXT:    or %s1, 12, (0)1
; CHECK-NEXT:    cmpu.w %s0, %s0, %s1
; CHECK-NEXT:    or %s1, 0, (0)1
; CHECK-NEXT:    cmov.w.lt %s1, (63)0, %s0
; CHECK-NEXT:    adds.w.zx %s0, %s1, (0)1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = icmp ult i32 %0, 12
  ret i1 %3
}

define zeroext i1 @setccule(i32, i32) {
; CHECK-LABEL: setccule:
; CHECK:       # %bb.0:
; CHECK-NEXT:    or %s1, 13, (0)1
; CHECK-NEXT:    cmpu.w %s0, %s0, %s1
; CHECK-NEXT:    or %s1, 0, (0)1
; CHECK-NEXT:    cmov.w.lt %s1, (63)0, %s0
; CHECK-NEXT:    adds.w.zx %s0, %s1, (0)1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = icmp ule i32 %0, 12
  ret i1 %3
}

define zeroext i1 @setccsgt(i32, i32) {
; CHECK-LABEL: setccsgt:
; CHECK:       # %bb.0:
; CHECK-NEXT:    or %s1, 12, (0)1
; CHECK-NEXT:    cmps.w.sx %s0, %s0, %s1
; CHECK-NEXT:    or %s1, 0, (0)1
; CHECK-NEXT:    cmov.w.gt %s1, (63)0, %s0
; CHECK-NEXT:    adds.w.zx %s0, %s1, (0)1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = icmp sgt i32 %0, 12
  ret i1 %3
}

define zeroext i1 @setccsge(i32, i32) {
; CHECK-LABEL: setccsge:
; CHECK:       # %bb.0:
; CHECK-NEXT:    or %s1, 11, (0)1
; CHECK-NEXT:    cmps.w.sx %s0, %s0, %s1
; CHECK-NEXT:    or %s1, 0, (0)1
; CHECK-NEXT:    cmov.w.gt %s1, (63)0, %s0
; CHECK-NEXT:    adds.w.zx %s0, %s1, (0)1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = icmp sge i32 %0, 12
  ret i1 %3
}

define zeroext i1 @setccslt(i32, i32) {
; CHECK-LABEL: setccslt:
; CHECK:       # %bb.0:
; CHECK-NEXT:    or %s1, 12, (0)1
; CHECK-NEXT:    cmps.w.sx %s0, %s0, %s1
; CHECK-NEXT:    or %s1, 0, (0)1
; CHECK-NEXT:    cmov.w.lt %s1, (63)0, %s0
; CHECK-NEXT:    adds.w.zx %s0, %s1, (0)1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = icmp slt i32 %0, 12
  ret i1 %3
}

define zeroext i1 @setccsle(i32, i32) {
; CHECK-LABEL: setccsle:
; CHECK:       # %bb.0:
; CHECK-NEXT:    or %s1, 13, (0)1
; CHECK-NEXT:    cmps.w.sx %s0, %s0, %s1
; CHECK-NEXT:    or %s1, 0, (0)1
; CHECK-NEXT:    cmov.w.lt %s1, (63)0, %s0
; CHECK-NEXT:    adds.w.zx %s0, %s1, (0)1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = icmp sle i32 %0, 12
  ret i1 %3
}
