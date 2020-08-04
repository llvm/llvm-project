; RUN: llc < %s -mtriple=ve-unknown-unknown | FileCheck %s

define signext i8 @func1(i8 signext %a, i8 signext %b) {
; CHECK-LABEL: func1:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    muls.w.sx %s0, %s1, %s0
; CHECK-NEXT:    sll %s0, %s0, 56
; CHECK-NEXT:    sra.l %s0, %s0, 56
; CHECK-NEXT:    or %s11, 0, %s9
  %r = mul i8 %b, %a
  ret i8 %r
}

define signext i16 @func2(i16 signext %a, i16 signext %b) {
; CHECK-LABEL: func2:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    muls.w.sx %s0, %s1, %s0
; CHECK-NEXT:    sll %s0, %s0, 48
; CHECK-NEXT:    sra.l %s0, %s0, 48
; CHECK-NEXT:    or %s11, 0, %s9
  %r = mul i16 %b, %a
  ret i16 %r
}

define i32 @func3(i32 %a, i32 %b) {
; CHECK-LABEL: func3:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    muls.w.sx %s0, %s1, %s0
; CHECK-NEXT:    or %s11, 0, %s9
  %r = mul nsw i32 %b, %a
  ret i32 %r
}

define i64 @func4(i64 %a, i64 %b) {
; CHECK-LABEL: func4:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    muls.l %s0, %s1, %s0
; CHECK-NEXT:    or %s11, 0, %s9
  %r = mul nsw i64 %b, %a
  ret i64 %r
}

define zeroext i8 @func5(i8 zeroext %a, i8 zeroext %b) {
; CHECK-LABEL: func5:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    muls.w.sx %s0, %s1, %s0
; CHECK-NEXT:    and %s0, %s0, (56)0
; CHECK-NEXT:    or %s11, 0, %s9
  %r = mul i8 %b, %a
  ret i8 %r
}

define zeroext i16 @func6(i16 zeroext %a, i16 zeroext %b) {
; CHECK-LABEL: func6:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    muls.w.sx %s0, %s1, %s0
; CHECK-NEXT:    and %s0, %s0, (48)0
; CHECK-NEXT:    or %s11, 0, %s9
  %r = mul i16 %b, %a
  ret i16 %r
}

define i32 @func7(i32 %a, i32 %b) {
; CHECK-LABEL: func7:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    muls.w.sx %s0, %s1, %s0
; CHECK-NEXT:    or %s11, 0, %s9
  %r = mul i32 %b, %a
  ret i32 %r
}

define i64 @func8(i64 %a, i64 %b) {
; CHECK-LABEL: func8:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    muls.l %s0, %s1, %s0
; CHECK-NEXT:    or %s11, 0, %s9
  %r = mul i64 %b, %a
  ret i64 %r
}

define signext i8 @func9(i8 signext %a) {
; CHECK-LABEL: func9:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    muls.w.sx %s0, 5, %s0
; CHECK-NEXT:    sll %s0, %s0, 56
; CHECK-NEXT:    sra.l %s0, %s0, 56
; CHECK-NEXT:    or %s11, 0, %s9
  %r = mul i8 %a, 5
  ret i8 %r
}

define signext i16 @func10(i16 signext %a) {
; CHECK-LABEL: func10:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    muls.w.sx %s0, 5, %s0
; CHECK-NEXT:    sll %s0, %s0, 48
; CHECK-NEXT:    sra.l %s0, %s0, 48
; CHECK-NEXT:    or %s11, 0, %s9
  %r = mul i16 %a, 5
  ret i16 %r
}

define i32 @func11(i32 %a) {
; CHECK-LABEL: func11:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    muls.w.sx %s0, 5, %s0
; CHECK-NEXT:    or %s11, 0, %s9
  %r = mul nsw i32 %a, 5
  ret i32 %r
}

define i64 @func12(i64 %a) {
; CHECK-LABEL: func12:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    muls.l %s0, 5, %s0
; CHECK-NEXT:    or %s11, 0, %s9
  %r = mul nsw i64 %a, 5
  ret i64 %r
}

define zeroext i8 @func13(i8 zeroext %a) {
; CHECK-LABEL: func13:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    muls.w.sx %s0, 5, %s0
; CHECK-NEXT:    and %s0, %s0, (56)0
; CHECK-NEXT:    or %s11, 0, %s9
  %r = mul i8 %a, 5
  ret i8 %r
}

define zeroext i16 @func14(i16 zeroext %a) {
; CHECK-LABEL: func14:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    muls.w.sx %s0, 5, %s0
; CHECK-NEXT:    and %s0, %s0, (48)0
; CHECK-NEXT:    or %s11, 0, %s9
  %r = mul i16 %a, 5
  ret i16 %r
}

define i32 @func15(i32 %a) {
; CHECK-LABEL: func15:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    muls.w.sx %s0, 5, %s0
; CHECK-NEXT:    or %s11, 0, %s9
  %r = mul i32 %a, 5
  ret i32 %r
}

define i64 @func16(i64 %a) {
; CHECK-LABEL: func16:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    muls.l %s0, 5, %s0
; CHECK-NEXT:    or %s11, 0, %s9
  %r = mul i64 %a, 5
  ret i64 %r
}

define i32 @func17(i32 %a) {
; CHECK-LABEL: func17:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    sla.w.sx %s0, %s0, 31
; CHECK-NEXT:    or %s11, 0, %s9
  %r = shl i32 %a, 31
  ret i32 %r
}

define i64 @func18(i64 %a) {
; CHECK-LABEL: func18:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    sll %s0, %s0, 31
; CHECK-NEXT:    or %s11, 0, %s9
  %r = shl nsw i64 %a, 31
  ret i64 %r
}
