; RUN: llc < %s -mtriple=ve-unknown-unknown | FileCheck %s

; Function Attrs: norecurse nounwind readnone
define i64 @divi64(i64 %a, i64 %b) {
; CHECK-LABEL: divi64:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    divs.l %s0, %s0, %s1
; CHECK-NEXT:    or %s11, 0, %s9
  %r = sdiv i64 %a, %b
  ret i64 %r
}

; Function Attrs: norecurse nounwind readnone
define i32 @divi32(i32 %a, i32 %b) {
; CHECK-LABEL: divi32:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    divs.w.sx %s0, %s0, %s1
; CHECK-NEXT:    or %s11, 0, %s9
  %r = sdiv i32 %a, %b
  ret i32 %r
}

; Function Attrs: norecurse nounwind readnone
define i64 @divu64(i64 %a, i64 %b) {
; CHECK-LABEL: divu64:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    divu.l %s0, %s0, %s1
; CHECK-NEXT:    or %s11, 0, %s9
  %r = udiv i64 %a, %b
  ret i64 %r
}

; Function Attrs: norecurse nounwind readnone
define i32 @divu32(i32 %a, i32 %b) {
; CHECK-LABEL: divu32:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    divu.w %s0, %s0, %s1
; CHECK-NEXT:    or %s11, 0, %s9
  %r = udiv i32 %a, %b
  ret i32 %r
}

; Function Attrs: norecurse nounwind readnone
define signext i16 @divi16(i16 signext %a, i16 signext %b) {
; CHECK-LABEL: divi16:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    divs.w.sx %s0, %s0, %s1
; CHECK-NEXT:    sla.w.sx %s0, %s0, 16
; CHECK-NEXT:    sra.w.sx %s0, %s0, 16
; CHECK-NEXT:    or %s11, 0, %s9
  %a32 = sext i16 %a to i32
  %b32 = sext i16 %b to i32
  %r32 = sdiv i32 %a32, %b32
  %r = trunc i32 %r32 to i16
  ret i16 %r
}

; Function Attrs: norecurse nounwind readnone
define zeroext i16 @divu16(i16 zeroext %a, i16 zeroext %b) {
; CHECK-LABEL: divu16:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    divu.w %s0, %s0, %s1
; CHECK-NEXT:    or %s11, 0, %s9
  %r = udiv i16 %a, %b
  ret i16 %r
}

; Function Attrs: norecurse nounwind readnone
define signext i8 @divi8(i8 signext %a, i8 signext %b) {
; CHECK-LABEL: divi8:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    divs.w.sx %s0, %s0, %s1
; CHECK-NEXT:    sla.w.sx %s0, %s0, 24
; CHECK-NEXT:    sra.w.sx %s0, %s0, 24
; CHECK-NEXT:    or %s11, 0, %s9
  %a32 = sext i8 %a to i32
  %b32 = sext i8 %b to i32
  %r32 = sdiv i32 %a32, %b32
  %r = trunc i32 %r32 to i8
  ret i8 %r
}

; Function Attrs: norecurse nounwind readnone
define zeroext i8 @divu8(i8 zeroext %a, i8 zeroext %b) {
; CHECK-LABEL: divu8:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    divu.w %s0, %s0, %s1
; CHECK-NEXT:    or %s11, 0, %s9
  %r = udiv i8 %a, %b
  ret i8 %r
}

; Function Attrs: norecurse nounwind readnone
define i64 @divi64ri(i64 %a, i64 %b) {
; CHECK-LABEL: divi64ri:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    divs.l %s0, %s0, (62)0
; CHECK-NEXT:    or %s11, 0, %s9
  %r = sdiv i64 %a, 3
  ret i64 %r
}

; Function Attrs: norecurse nounwind readnone
define i32 @divi32ri(i32 %a, i32 %b) {
; CHECK-LABEL: divi32ri:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    divs.w.sx %s0, %s0, (62)0
; CHECK-NEXT:    or %s11, 0, %s9
  %r = sdiv i32 %a, 3
  ret i32 %r
}

; Function Attrs: norecurse nounwind readnone
define i64 @divu64ri(i64 %a, i64 %b) {
; CHECK-LABEL: divu64ri:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    divu.l %s0, %s0, (62)0
; CHECK-NEXT:    or %s11, 0, %s9
  %r = udiv i64 %a, 3
  ret i64 %r
}

; Function Attrs: norecurse nounwind readnone
define i32 @divu32ri(i32 %a, i32 %b) {
; CHECK-LABEL: divu32ri:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    divu.w %s0, %s0, (62)0
; CHECK-NEXT:    or %s11, 0, %s9
  %r = udiv i32 %a, 3
  ret i32 %r
}

; Function Attrs: norecurse nounwind readnone
define i64 @divi64li(i64 %a, i64 %b) {
; CHECK-LABEL: divi64li:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    divs.l %s0, 3, %s1
; CHECK-NEXT:    or %s11, 0, %s9
  %r = sdiv i64 3, %b
  ret i64 %r
}

; Function Attrs: norecurse nounwind readnone
define i32 @divi32li(i32 %a, i32 %b) {
; CHECK-LABEL: divi32li:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    divs.w.sx %s0, 3, %s1
; CHECK-NEXT:    or %s11, 0, %s9
  %r = sdiv i32 3, %b
  ret i32 %r
}

; Function Attrs: norecurse nounwind readnone
define i64 @divu64li(i64 %a, i64 %b) {
; CHECK-LABEL: divu64li:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    divu.l %s0, 3, %s1
; CHECK-NEXT:    or %s11, 0, %s9
  %r = udiv i64 3, %b
  ret i64 %r
}

; Function Attrs: norecurse nounwind readnone
define i32 @divu32li(i32 %a, i32 %b) {
; CHECK-LABEL: divu32li:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    divu.w %s0, 3, %s1
; CHECK-NEXT:    or %s11, 0, %s9
  %r = udiv i32 3, %b
  ret i32 %r
}
