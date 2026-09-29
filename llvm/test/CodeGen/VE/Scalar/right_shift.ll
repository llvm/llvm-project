; RUN: llc < %s -mtriple=ve-unknown-unknown | FileCheck %s

define signext i8 @func1(i8 signext %0, i8 signext %1) {
; CHECK-LABEL: func1:
; CHECK:       # %bb.0:
; CHECK-NEXT:    sra.w.sx %s0, %s0, %s1
; CHECK-NEXT:    adds.w.sx %s0, %s0, (0)1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = sext i8 %0 to i32
  %4 = sext i8 %1 to i32
  %5 = ashr i32 %3, %4
  %6 = trunc i32 %5 to i8
  ret i8 %6
}

define signext i16 @func2(i16 signext %0, i16 signext %1) {
; CHECK-LABEL: func2:
; CHECK:       # %bb.0:
; CHECK-NEXT:    sra.w.sx %s0, %s0, %s1
; CHECK-NEXT:    adds.w.sx %s0, %s0, (0)1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = sext i16 %0 to i32
  %4 = sext i16 %1 to i32
  %5 = ashr i32 %3, %4
  %6 = trunc i32 %5 to i16
  ret i16 %6
}

define i32 @func3(i32 %0, i32 %1) {
; CHECK-LABEL: func3:
; CHECK:       # %bb.0:
; CHECK-NEXT:    sra.w.sx %s0, %s0, %s1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = ashr i32 %0, %1
  ret i32 %3
}

define i64 @func4(i64 %0, i64 %1) {
; CHECK-LABEL: func4:
; CHECK:       # %bb.0:
; CHECK-NEXT:    sra.l %s0, %s0, %s1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = ashr i64 %0, %1
  ret i64 %3
}

define zeroext i8 @func7(i8 zeroext %0, i8 zeroext %1) {
; CHECK-LABEL: func7:
; CHECK:       # %bb.0:
; CHECK-NEXT:    and %s0, %s0, (32)0
; CHECK-NEXT:    srl %s0, %s0, %s1
; CHECK-NEXT:    adds.w.zx %s0, %s0, (0)1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = zext i8 %0 to i32
  %4 = zext i8 %1 to i32
  %5 = lshr i32 %3, %4
  %6 = trunc i32 %5 to i8
  ret i8 %6
}

define zeroext i16 @func8(i16 zeroext %0, i16 zeroext %1) {
; CHECK-LABEL: func8:
; CHECK:       # %bb.0:
; CHECK-NEXT:    and %s0, %s0, (32)0
; CHECK-NEXT:    srl %s0, %s0, %s1
; CHECK-NEXT:    adds.w.zx %s0, %s0, (0)1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = zext i16 %0 to i32
  %4 = zext i16 %1 to i32
  %5 = lshr i32 %3, %4
  %6 = trunc i32 %5 to i16
  ret i16 %6
}

define i32 @func9(i32 %0, i32 %1) {
; CHECK-LABEL: func9:
; CHECK:       # %bb.0:
; CHECK-NEXT:    and %s0, %s0, (32)0
; CHECK-NEXT:    srl %s0, %s0, %s1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = lshr i32 %0, %1
  ret i32 %3
}

define i64 @func10(i64 %0, i64 %1) {
; CHECK-LABEL: func10:
; CHECK:       # %bb.0:
; CHECK-NEXT:    srl %s0, %s0, %s1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = lshr i64 %0, %1
  ret i64 %3
}

define signext i8 @func12(i8 signext %0) {
; CHECK-LABEL: func12:
; CHECK:       # %bb.0:
; CHECK-NEXT:    sra.w.sx %s0, %s0, 5
; CHECK-NEXT:    adds.w.sx %s0, %s0, (0)1
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = ashr i8 %0, 5
  ret i8 %2
}

define signext i16 @func13(i16 signext %0) {
; CHECK-LABEL: func13:
; CHECK:       # %bb.0:
; CHECK-NEXT:    sra.w.sx %s0, %s0, 5
; CHECK-NEXT:    adds.w.sx %s0, %s0, (0)1
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = ashr i16 %0, 5
  ret i16 %2
}

define i32 @func14(i32 %0) {
; CHECK-LABEL: func14:
; CHECK:       # %bb.0:
; CHECK-NEXT:    sra.w.sx %s0, %s0, 5
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = ashr i32 %0, 5
  ret i32 %2
}

define i64 @func15(i64 %0) {
; CHECK-LABEL: func15:
; CHECK:       # %bb.0:
; CHECK-NEXT:    sra.l %s0, %s0, 5
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = ashr i64 %0, 5
  ret i64 %2
}

define zeroext i8 @func17(i8 zeroext %0) {
; CHECK-LABEL: func17:
; CHECK:       # %bb.0:
; CHECK-NEXT:    and %s0, %s0, (32)0
; CHECK-NEXT:    srl %s0, %s0, 5
; CHECK-NEXT:    adds.w.zx %s0, %s0, (0)1
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = lshr i8 %0, 5
  ret i8 %2
}

define zeroext i16 @func18(i16 zeroext %0) {
; CHECK-LABEL: func18:
; CHECK:       # %bb.0:
; CHECK-NEXT:    and %s0, %s0, (32)0
; CHECK-NEXT:    srl %s0, %s0, 5
; CHECK-NEXT:    adds.w.zx %s0, %s0, (0)1
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = lshr i16 %0, 5
  ret i16 %2
}

define i32 @func19(i32 %0) {
; CHECK-LABEL: func19:
; CHECK:       # %bb.0:
; CHECK-NEXT:    and %s0, %s0, (32)0
; CHECK-NEXT:    srl %s0, %s0, 5
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = lshr i32 %0, 5
  ret i32 %2
}

define i64 @func20(i64 %0) {
; CHECK-LABEL: func20:
; CHECK:       # %bb.0:
; CHECK-NEXT:    srl %s0, %s0, 5
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = lshr i64 %0, 5
  ret i64 %2
}

