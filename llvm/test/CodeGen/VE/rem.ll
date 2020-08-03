; RUN: llc < %s -mtriple=ve-unknown-unknown | FileCheck %s

; Function Attrs: norecurse nounwind readnone
define i64 @remi64(i64 %a, i64 %b) {
; CHECK-LABEL: remi64:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    divs.l %s2, %s0, %s1
; CHECK-NEXT:    muls.l %s1, %s2, %s1
; CHECK-NEXT:    subs.l %s0, %s0, %s1
; CHECK-NEXT:    or %s11, 0, %s9
  %r = srem i64 %a, %b
  ret i64 %r
}

; Function Attrs: norecurse nounwind readnone
define i32 @remi32(i32 %a, i32 %b) {
; CHECK-LABEL: remi32:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    adds.w.sx %s1, %s1, (0)1
; CHECK-NEXT:    adds.w.sx %s0, %s0, (0)1
; CHECK-NEXT:    divs.w.sx %s2, %s0, %s1
; CHECK-NEXT:    muls.w.sx %s1, %s2, %s1
; CHECK-NEXT:    subs.w.sx %s0, %s0, %s1
; CHECK-NEXT:    or %s11, 0, %s9
  %r = srem i32 %a, %b
  ret i32 %r
}

; Function Attrs: norecurse nounwind readnone
define i64 @remu64(i64 %a, i64 %b) {
; CHECK-LABEL: remu64:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    divu.l %s2, %s0, %s1
; CHECK-NEXT:    muls.l %s1, %s2, %s1
; CHECK-NEXT:    subs.l %s0, %s0, %s1
; CHECK-NEXT:    or %s11, 0, %s9
  %r = urem i64 %a, %b
  ret i64 %r
}

; Function Attrs: norecurse nounwind readnone
define i32 @remu32(i32 %a, i32 %b) {
; CHECK-LABEL: remu32:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    adds.w.sx %s1, %s1, (0)1
; CHECK-NEXT:    adds.w.sx %s0, %s0, (0)1
; CHECK-NEXT:    divu.w %s2, %s0, %s1
; CHECK-NEXT:    muls.w.sx %s1, %s2, %s1
; CHECK-NEXT:    subs.w.sx %s0, %s0, %s1
; CHECK-NEXT:    or %s11, 0, %s9
  %r = urem i32 %a, %b
  ret i32 %r
}

; Function Attrs: norecurse nounwind readnone
define signext i16 @remi16(i16 signext %a, i16 signext %b) {
; CHECK-LABEL: remi16:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    adds.w.sx %s1, %s1, (0)1
; CHECK-NEXT:    adds.w.sx %s0, %s0, (0)1
; CHECK-NEXT:    divs.w.sx %s2, %s0, %s1
; CHECK-NEXT:    muls.w.sx %s1, %s2, %s1
; CHECK-NEXT:    subs.w.sx %s0, %s0, %s1
; CHECK-NEXT:    sll %s0, %s0, 48
; CHECK-NEXT:    sra.l %s0, %s0, 48
; CHECK-NEXT:    or %s11, 0, %s9
  %a32 = sext i16 %a to i32
  %b32 = sext i16 %b to i32
  %r32 = srem i32 %a32, %b32
  %r = trunc i32 %r32 to i16
  ret i16 %r
}

; Function Attrs: norecurse nounwind readnone
define zeroext i16 @remu16(i16 zeroext %a, i16 zeroext %b) {
; CHECK-LABEL: remu16:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    adds.w.sx %s1, %s1, (0)1
; CHECK-NEXT:    adds.w.sx %s0, %s0, (0)1
; CHECK-NEXT:    divu.w %s2, %s0, %s1
; CHECK-NEXT:    muls.w.sx %s1, %s2, %s1
; CHECK-NEXT:    subs.w.sx %s0, %s0, %s1
; CHECK-NEXT:    adds.w.zx %s0, %s0, (0)1
; CHECK-NEXT:    or %s11, 0, %s9
  %r = urem i16 %a, %b
  ret i16 %r
}

; Function Attrs: norecurse nounwind readnone
define signext i8 @remi8(i8 signext %a, i8 signext %b) {
; CHECK-LABEL: remi8:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    adds.w.sx %s1, %s1, (0)1
; CHECK-NEXT:    adds.w.sx %s0, %s0, (0)1
; CHECK-NEXT:    divs.w.sx %s2, %s0, %s1
; CHECK-NEXT:    muls.w.sx %s1, %s2, %s1
; CHECK-NEXT:    subs.w.sx %s0, %s0, %s1
; CHECK-NEXT:    sll %s0, %s0, 56
; CHECK-NEXT:    sra.l %s0, %s0, 56
; CHECK-NEXT:    or %s11, 0, %s9
  %a32 = sext i8 %a to i32
  %b32 = sext i8 %b to i32
  %r32 = srem i32 %a32, %b32
  %r = trunc i32 %r32 to i8
  ret i8 %r
}

; Function Attrs: norecurse nounwind readnone
define zeroext i8 @remu8(i8 zeroext %a, i8 zeroext %b) {
; CHECK-LABEL: remu8:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    adds.w.sx %s1, %s1, (0)1
; CHECK-NEXT:    adds.w.sx %s0, %s0, (0)1
; CHECK-NEXT:    divu.w %s2, %s0, %s1
; CHECK-NEXT:    muls.w.sx %s1, %s2, %s1
; CHECK-NEXT:    subs.w.sx %s0, %s0, %s1
; CHECK-NEXT:    adds.w.zx %s0, %s0, (0)1
; CHECK-NEXT:    or %s11, 0, %s9
  %r = urem i8 %a, %b
  ret i8 %r
}

; Function Attrs: norecurse nounwind readnone
define i64 @remi64ri(i64 %a, i64 %b) {
; CHECK-LABEL: remi64ri:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    divs.l %s1, %s0, (62)0
; CHECK-NEXT:    muls.l %s1, 3, %s1
; CHECK-NEXT:    subs.l %s0, %s0, %s1
; CHECK-NEXT:    or %s11, 0, %s9
  %r = srem i64 %a, 3
  ret i64 %r
}

; Function Attrs: norecurse nounwind readnone
define i32 @remi32ri(i32 %a, i32 %b) {
; CHECK-LABEL: remi32ri:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    adds.w.sx %s0, %s0, (0)1
; CHECK-NEXT:    divs.w.sx %s1, %s0, (62)0
; CHECK-NEXT:    muls.w.sx %s1, 3, %s1
; CHECK-NEXT:    subs.w.sx %s0, %s0, %s1
; CHECK-NEXT:    or %s11, 0, %s9
  %r = srem i32 %a, 3
  ret i32 %r
}

; Function Attrs: norecurse nounwind readnone
define i64 @remu64ri(i64 %a, i64 %b) {
; CHECK-LABEL: remu64ri:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    divu.l %s1, %s0, (62)0
; CHECK-NEXT:    muls.l %s1, 3, %s1
; CHECK-NEXT:    subs.l %s0, %s0, %s1
; CHECK-NEXT:    or %s11, 0, %s9
  %r = urem i64 %a, 3
  ret i64 %r
}

; Function Attrs: norecurse nounwind readnone
define i32 @remu32ri(i32 %a, i32 %b) {
; CHECK-LABEL: remu32ri:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    adds.w.sx %s0, %s0, (0)1
; CHECK-NEXT:    divu.w %s1, %s0, (62)0
; CHECK-NEXT:    muls.w.sx %s1, 3, %s1
; CHECK-NEXT:    subs.w.sx %s0, %s0, %s1
; CHECK-NEXT:    or %s11, 0, %s9
  %r = urem i32 %a, 3
  ret i32 %r
}

; Function Attrs: norecurse nounwind readnone
define i64 @remi64li(i64 %a, i64 %b) {
; CHECK-LABEL: remi64li:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    divs.l %s0, 3, %s1
; CHECK-NEXT:    muls.l %s0, %s0, %s1
; CHECK-NEXT:    subs.l %s0, 3, %s0
; CHECK-NEXT:    or %s11, 0, %s9
  %r = srem i64 3, %b
  ret i64 %r
}

; Function Attrs: norecurse nounwind readnone
define i32 @remi32li(i32 %a, i32 %b) {
; CHECK-LABEL: remi32li:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    adds.w.sx %s0, %s1, (0)1
; CHECK-NEXT:    divs.w.sx %s1, 3, %s0
; CHECK-NEXT:    muls.w.sx %s0, %s1, %s0
; CHECK-NEXT:    subs.w.sx %s0, 3, %s0
; CHECK-NEXT:    or %s11, 0, %s9
  %r = srem i32 3, %b
  ret i32 %r
}

; Function Attrs: norecurse nounwind readnone
define i64 @remu64li(i64 %a, i64 %b) {
; CHECK-LABEL: remu64li:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    divu.l %s0, 3, %s1
; CHECK-NEXT:    muls.l %s0, %s0, %s1
; CHECK-NEXT:    subs.l %s0, 3, %s0
; CHECK-NEXT:    or %s11, 0, %s9
  %r = urem i64 3, %b
  ret i64 %r
}

; Function Attrs: norecurse nounwind readnone
define i32 @remu32li(i32 %a, i32 %b) {
; CHECK-LABEL: remu32li:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    adds.w.sx %s0, %s1, (0)1
; CHECK-NEXT:    divu.w %s1, 3, %s0
; CHECK-NEXT:    muls.w.sx %s0, %s1, %s0
; CHECK-NEXT:    subs.w.sx %s0, 3, %s0
; CHECK-NEXT:    or %s11, 0, %s9
  %r = urem i32 3, %b
  ret i32 %r
}
