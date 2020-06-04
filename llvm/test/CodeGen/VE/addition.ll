; RUN: llc < %s -mtriple=ve-unknown-unknown | FileCheck %s

define signext i8 @func1(i8 signext %0, i8 signext %1) {
; CHECK-LABEL: func1:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    adds.w.sx %s0, %s1, %s0
; CHECK-NEXT:    sla.w.sx %s0, %s0, 24
; CHECK-NEXT:    sra.w.sx %s0, %s0, 24
; CHECK-NEXT:    or %s11, 0, %s9
  %3 = add i8 %1, %0
  ret i8 %3
}

define signext i16 @func2(i16 signext %0, i16 signext %1) {
; CHECK-LABEL: func2:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    adds.w.sx %s0, %s1, %s0
; CHECK-NEXT:    sla.w.sx %s0, %s0, 16
; CHECK-NEXT:    sra.w.sx %s0, %s0, 16
; CHECK-NEXT:    or %s11, 0, %s9
  %3 = add i16 %1, %0
  ret i16 %3
}

define i32 @func3(i32 %0, i32 %1) {
; CHECK-LABEL: func3:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    adds.w.sx %s0, %s1, %s0
; CHECK-NEXT:    or %s11, 0, %s9
  %3 = add nsw i32 %1, %0
  ret i32 %3
}

define i64 @func4(i64 %0, i64 %1) {
; CHECK-LABEL: func4:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    adds.l %s0, %s1, %s0
; CHECK-NEXT:    or %s11, 0, %s9
  %3 = add nsw i64 %1, %0
  ret i64 %3
}

define zeroext i8 @func6(i8 zeroext %0, i8 zeroext %1) {
; CHECK-LABEL: func6:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    adds.w.sx %s0, %s1, %s0
; CHECK-NEXT:    and %s0, %s0, (56)0
; CHECK-NEXT:    or %s11, 0, %s9
  %3 = add i8 %1, %0
  ret i8 %3
}

define zeroext i16 @func7(i16 zeroext %0, i16 zeroext %1) {
; CHECK-LABEL: func7:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    adds.w.sx %s0, %s1, %s0
; CHECK-NEXT:    and %s0, %s0, (48)0
; CHECK-NEXT:    or %s11, 0, %s9
  %3 = add i16 %1, %0
  ret i16 %3
}

define i32 @func8(i32 %0, i32 %1) {
; CHECK-LABEL: func8:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    adds.w.sx %s0, %s1, %s0
; CHECK-NEXT:    or %s11, 0, %s9
  %3 = add i32 %1, %0
  ret i32 %3
}

define i64 @func9(i64 %0, i64 %1) {
; CHECK-LABEL: func9:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    adds.l %s0, %s1, %s0
; CHECK-NEXT:    or %s11, 0, %s9
  %3 = add i64 %1, %0
  ret i64 %3
}

define signext i8 @func13(i8 signext %0) {
; CHECK-LABEL: func13:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    adds.w.sx %s0, 5, %s0
; CHECK-NEXT:    sla.w.sx %s0, %s0, 24
; CHECK-NEXT:    sra.w.sx %s0, %s0, 24
; CHECK-NEXT:    or %s11, 0, %s9
  %2 = add i8 %0, 5
  ret i8 %2
}

define signext i16 @func14(i16 signext %0) {
; CHECK-LABEL: func14:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    adds.w.sx %s0, 5, %s0
; CHECK-NEXT:    sla.w.sx %s0, %s0, 16
; CHECK-NEXT:    sra.w.sx %s0, %s0, 16
; CHECK-NEXT:    or %s11, 0, %s9
  %2 = add i16 %0, 5
  ret i16 %2
}

define i32 @func15(i32 %0) {
; CHECK-LABEL: func15:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    adds.w.sx %s0, 5, %s0
; CHECK-NEXT:    or %s11, 0, %s9
  %2 = add nsw i32 %0, 5
  ret i32 %2
}

define i64 @func16(i64 %0) {
; CHECK-LABEL: func16:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    lea %s0, 5(, %s0)
; CHECK-NEXT:    or %s11, 0, %s9
  %2 = add nsw i64 %0, 5
  ret i64 %2
}

define zeroext i8 @func18(i8 zeroext %0) {
; CHECK-LABEL: func18:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    adds.w.sx %s0, 5, %s0
; CHECK-NEXT:    and %s0, %s0, (56)0
; CHECK-NEXT:    or %s11, 0, %s9
  %2 = add i8 %0, 5
  ret i8 %2
}

define zeroext i16 @func19(i16 zeroext %0) {
; CHECK-LABEL: func19:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    adds.w.sx %s0, 5, %s0
; CHECK-NEXT:    and %s0, %s0, (48)0
; CHECK-NEXT:    or %s11, 0, %s9
  %2 = add i16 %0, 5
  ret i16 %2
}

define i32 @func20(i32 %0) {
; CHECK-LABEL: func20:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    adds.w.sx %s0, 5, %s0
; CHECK-NEXT:    or %s11, 0, %s9
  %2 = add i32 %0, 5
  ret i32 %2
}

define i64 @func21(i64 %0) {
; CHECK-LABEL: func21:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    lea %s0, 5(, %s0)
; CHECK-NEXT:    or %s11, 0, %s9
  %2 = add i64 %0, 5
  ret i64 %2
}

define i32 @func25(i32 %0) {
; CHECK-LABEL: func25:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    xor %s0, %s0, (33)1
; CHECK-NEXT:    or %s11, 0, %s9
  %2 = xor i32 %0, -2147483648
  ret i32 %2
}

define i64 @func26(i64 %0) {
; CHECK-LABEL: func26:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    lea %s1, -2147483648
; CHECK-NEXT:    and %s1, %s1, (32)0
; CHECK-NEXT:    adds.l %s0, %s0, %s1
; CHECK-NEXT:    or %s11, 0, %s9
  %2 = add nsw i64 %0, 2147483648
  ret i64 %2
}

