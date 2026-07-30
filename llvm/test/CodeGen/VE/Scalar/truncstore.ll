; RUN: llc < %s -mtriple=ve-unknown-unknown | FileCheck %s

define void @func0(i1 signext %p, ptr %a) {
; CHECK-LABEL: func0:
; CHECK:       # %bb.0:
; CHECK-NEXT:    st1b %s0, (, %s1)
; CHECK-NEXT:    b.l.t (, %s10)
  %p.conv = sext i1 %p to i8
  store i8 %p.conv, ptr %a, align 2
  ret void
}

define void @func1(i8 signext %p, ptr %a) {
; CHECK-LABEL: func1:
; CHECK:       # %bb.0:
; CHECK-NEXT:    st2b %s0, (, %s1)
; CHECK-NEXT:    b.l.t (, %s10)
  %p.conv = sext i8 %p to i16
  store i16 %p.conv, ptr %a, align 2
  ret void
}

define void @func2(i8 signext %p, ptr %a) {
; CHECK-LABEL: func2:
; CHECK:       # %bb.0:
; CHECK-NEXT:    stl %s0, (, %s1)
; CHECK-NEXT:    b.l.t (, %s10)
  %p.conv = sext i8 %p to i32
  store i32 %p.conv, ptr %a, align 4
  ret void
}

define void @func3(i8 signext %p, ptr %a) {
; CHECK-LABEL: func3:
; CHECK:       # %bb.0:
; CHECK-NEXT:    st %s0, (, %s1)
; CHECK-NEXT:    b.l.t (, %s10)
  %p.conv = sext i8 %p to i64
  store i64 %p.conv, ptr %a, align 8
  ret void
}

define void @func5(i16 signext %p, ptr %a) {
; CHECK-LABEL: func5:
; CHECK:       # %bb.0:
; CHECK-NEXT:    stl %s0, (, %s1)
; CHECK-NEXT:    b.l.t (, %s10)
  %p.conv = sext i16 %p to i32
  store i32 %p.conv, ptr %a, align 4
  ret void
}

define void @func6(i16 signext %p, ptr %a) {
; CHECK-LABEL: func6:
; CHECK:       # %bb.0:
; CHECK-NEXT:    st %s0, (, %s1)
; CHECK-NEXT:    b.l.t (, %s10)
  %p.conv = sext i16 %p to i64
  store i64 %p.conv, ptr %a, align 8
  ret void
}

define void @func8(i32 %p, ptr %a) {
; CHECK-LABEL: func8:
; CHECK:       # %bb.0:
; CHECK-NEXT:    adds.w.sx %s0, %s0, (0)1
; CHECK-NEXT:    st %s0, (, %s1)
; CHECK-NEXT:    b.l.t (, %s10)
  %p.conv = sext i32 %p to i64
  store i64 %p.conv, ptr %a, align 8
  ret void
}
