; RUN: llc < %s -mtriple=ve-unknown-unknown | FileCheck %s

define signext i8 @func1(i8 signext %0, i8 signext %1) {
; CHECK-LABEL: func1:
; CHECK:       # %bb.0:
; CHECK-NEXT:    sla.w.sx %s0, %s0, %s1
; CHECK-NEXT:    sll %s0, %s0, 56
; CHECK-NEXT:    sra.l %s0, %s0, 56
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = sext i8 %0 to i32
  %4 = sext i8 %1 to i32
  %5 = shl i32 %3, %4
  %6 = trunc i32 %5 to i8
  ret i8 %6
}

define signext i16 @func2(i16 signext %0, i16 signext %1) {
; CHECK-LABEL: func2:
; CHECK:       # %bb.0:
; CHECK-NEXT:    sla.w.sx %s0, %s0, %s1
; CHECK-NEXT:    sll %s0, %s0, 48
; CHECK-NEXT:    sra.l %s0, %s0, 48
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = sext i16 %0 to i32
  %4 = sext i16 %1 to i32
  %5 = shl i32 %3, %4
  %6 = trunc i32 %5 to i16
  ret i16 %6
}

define i32 @func3(i32 %0, i32 %1) {
; CHECK-LABEL: func3:
; CHECK:       # %bb.0:
; CHECK-NEXT:    sla.w.sx %s0, %s0, %s1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = shl i32 %0, %1
  ret i32 %3
}

define i64 @func4(i64 %0, i64 %1) {
; CHECK-LABEL: func4:
; CHECK:       # %bb.0:
; CHECK-NEXT:    sll %s0, %s0, %s1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = shl i64 %0, %1
  ret i64 %3
}

define zeroext i8 @func6(i8 zeroext %0, i8 zeroext %1) {
; CHECK-LABEL: func6:
; CHECK:       # %bb.0:
; CHECK-NEXT:    sla.w.sx %s0, %s0, %s1
; CHECK-NEXT:    and %s0, %s0, (56)0
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = zext i8 %0 to i32
  %4 = zext i8 %1 to i32
  %5 = shl i32 %3, %4
  %6 = trunc i32 %5 to i8
  ret i8 %6
}

define zeroext i16 @func7(i16 zeroext %0, i16 zeroext %1) {
; CHECK-LABEL: func7:
; CHECK:       # %bb.0:
; CHECK-NEXT:    sla.w.sx %s0, %s0, %s1
; CHECK-NEXT:    and %s0, %s0, (48)0
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = zext i16 %0 to i32
  %4 = zext i16 %1 to i32
  %5 = shl i32 %3, %4
  %6 = trunc i32 %5 to i16
  ret i16 %6
}

define i32 @func8(i32 %0, i32 %1) {
; CHECK-LABEL: func8:
; CHECK:       # %bb.0:
; CHECK-NEXT:    sla.w.sx %s0, %s0, %s1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = shl i32 %0, %1
  ret i32 %3
}

define i64 @func9(i64 %0, i64 %1) {
; CHECK-LABEL: func9:
; CHECK:       # %bb.0:
; CHECK-NEXT:    sll %s0, %s0, %s1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = shl i64 %0, %1
  ret i64 %3
}

define signext i8 @func11(i8 signext %0) {
; CHECK-LABEL: func11:
; CHECK:       # %bb.0:
; CHECK-NEXT:    sla.w.sx %s0, %s0, 5
; CHECK-NEXT:    sll %s0, %s0, 56
; CHECK-NEXT:    sra.l %s0, %s0, 56
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = shl i8 %0, 5
  ret i8 %2
}

define signext i16 @func12(i16 signext %0) {
; CHECK-LABEL: func12:
; CHECK:       # %bb.0:
; CHECK-NEXT:    sla.w.sx %s0, %s0, 5
; CHECK-NEXT:    sll %s0, %s0, 48
; CHECK-NEXT:    sra.l %s0, %s0, 48
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = shl i16 %0, 5
  ret i16 %2
}

define i32 @func13(i32 %0) {
; CHECK-LABEL: func13:
; CHECK:       # %bb.0:
; CHECK-NEXT:    sla.w.sx %s0, %s0, 5
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = shl i32 %0, 5
  ret i32 %2
}

define i64 @func14(i64 %0) {
; CHECK-LABEL: func14:
; CHECK:       # %bb.0:
; CHECK-NEXT:    sll %s0, %s0, 5
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = shl i64 %0, 5
  ret i64 %2
}

define zeroext i8 @func16(i8 zeroext %0) {
; CHECK-LABEL: func16:
; CHECK:       # %bb.0:
; CHECK-NEXT:    sla.w.sx %s0, %s0, 5
; CHECK-NEXT:    lea %s1, 224
; CHECK-NEXT:    and %s0, %s0, %s1
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = shl i8 %0, 5
  ret i8 %2
}

define zeroext i16 @func17(i16 zeroext %0) {
; CHECK-LABEL: func17:
; CHECK:       # %bb.0:
; CHECK-NEXT:    sla.w.sx %s0, %s0, 5
; CHECK-NEXT:    lea %s1, 65504
; CHECK-NEXT:    and %s0, %s0, %s1
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = shl i16 %0, 5
  ret i16 %2
}

define i32 @func18(i32 %0) {
; CHECK-LABEL: func18:
; CHECK:       # %bb.0:
; CHECK-NEXT:    sla.w.sx %s0, %s0, 5
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = shl i32 %0, 5
  ret i32 %2
}

define i64 @func19(i64 %0) {
; CHECK-LABEL: func19:
; CHECK:       # %bb.0:
; CHECK-NEXT:    sll %s0, %s0, 5
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = shl i64 %0, 5
  ret i64 %2
}

