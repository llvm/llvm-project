; RUN: llc < %s -mtriple=ve-unknown-unknown | FileCheck %s

; Function Attrs: norecurse nounwind readnone
define i128 @divi128(i128, i128) {
; CHECK-LABEL: divi128:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    lea %s4, __divti3@lo
; CHECK-NEXT:    and %s4, %s4, (32)0
; CHECK-NEXT:    lea.sl %s12, __divti3@hi(, %s4)
; CHECK-NEXT:    bsic %s10, (, %s12)
; CHECK-NEXT:    or %s11, 0, %s9
  %3 = sdiv i128 %0, %1
  ret i128 %3
}

; Function Attrs: norecurse nounwind readnone
define i64 @divi64(i64 %a, i64 %b) {
; CHECK-LABEL: divi64:
; CHECK:       # %bb.0:
; CHECK-NEXT:    divs.l %s0, %s0, %s1
; CHECK-NEXT:    b.l.t (, %s10)
  %r = sdiv i64 %a, %b
  ret i64 %r
}

; Function Attrs: norecurse nounwind readnone
define signext i32 @divi32(i32 signext %a, i32 signext %b) {
; CHECK-LABEL: divi32:
; CHECK:       # %bb.0:
; CHECK-NEXT:    divs.w.sx %s0, %s0, %s1
; CHECK-NEXT:    adds.w.sx %s0, %s0, (0)1
; CHECK-NEXT:    b.l.t (, %s10)
  %r = sdiv i32 %a, %b
  ret i32 %r
}

; Function Attrs: norecurse nounwind readnone
define i128 @divu128(i128, i128) {
; CHECK-LABEL: divu128:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    lea %s4, __udivti3@lo
; CHECK-NEXT:    and %s4, %s4, (32)0
; CHECK-NEXT:    lea.sl %s12, __udivti3@hi(, %s4)
; CHECK-NEXT:    bsic %s10, (, %s12)
; CHECK-NEXT:    or %s11, 0, %s9
  %3 = udiv i128 %0, %1
  ret i128 %3
}

; Function Attrs: norecurse nounwind readnone
define i64 @divu64(i64 %a, i64 %b) {
; CHECK-LABEL: divu64:
; CHECK:       # %bb.0:
; CHECK-NEXT:    divu.l %s0, %s0, %s1
; CHECK-NEXT:    b.l.t (, %s10)
  %r = udiv i64 %a, %b
  ret i64 %r
}

; Function Attrs: norecurse nounwind readnone
define zeroext i32 @divu32(i32 zeroext %a, i32 zeroext %b) {
; CHECK-LABEL: divu32:
; CHECK:       # %bb.0:
; CHECK-NEXT:    divu.w %s0, %s0, %s1
; CHECK-NEXT:    adds.w.zx %s0, %s0, (0)1
; CHECK-NEXT:    b.l.t (, %s10)
  %r = udiv i32 %a, %b
  ret i32 %r
}

; Function Attrs: norecurse nounwind readnone
define signext i16 @divi16(i16 signext %a, i16 signext %b) {
; CHECK-LABEL: divi16:
; CHECK:       # %bb.0:
; CHECK-NEXT:    divs.w.sx %s0, %s0, %s1
; CHECK-NEXT:    sll %s0, %s0, 48
; CHECK-NEXT:    sra.l %s0, %s0, 48
; CHECK-NEXT:    b.l.t (, %s10)
  %a32 = sext i16 %a to i32
  %b32 = sext i16 %b to i32
  %r32 = sdiv i32 %a32, %b32
  %r = trunc i32 %r32 to i16
  ret i16 %r
}

; Function Attrs: norecurse nounwind readnone
define zeroext i16 @divu16(i16 zeroext %a, i16 zeroext %b) {
; CHECK-LABEL: divu16:
; CHECK:       # %bb.0:
; CHECK-NEXT:    divu.w %s0, %s0, %s1
; CHECK-NEXT:    adds.w.zx %s0, %s0, (0)1
; CHECK-NEXT:    b.l.t (, %s10)
  %r = udiv i16 %a, %b
  ret i16 %r
}

; Function Attrs: norecurse nounwind readnone
define signext i8 @divi8(i8 signext %a, i8 signext %b) {
; CHECK-LABEL: divi8:
; CHECK:       # %bb.0:
; CHECK-NEXT:    divs.w.sx %s0, %s0, %s1
; CHECK-NEXT:    sll %s0, %s0, 56
; CHECK-NEXT:    sra.l %s0, %s0, 56
; CHECK-NEXT:    b.l.t (, %s10)
  %a32 = sext i8 %a to i32
  %b32 = sext i8 %b to i32
  %r32 = sdiv i32 %a32, %b32
  %r = trunc i32 %r32 to i8
  ret i8 %r
}

; Function Attrs: norecurse nounwind readnone
define zeroext i8 @divu8(i8 zeroext %a, i8 zeroext %b) {
; CHECK-LABEL: divu8:
; CHECK:       # %bb.0:
; CHECK-NEXT:    divu.w %s0, %s0, %s1
; CHECK-NEXT:    adds.w.zx %s0, %s0, (0)1
; CHECK-NEXT:    b.l.t (, %s10)
  %r = udiv i8 %a, %b
  ret i8 %r
}

; Function Attrs: norecurse nounwind readnone
define i128 @divi128ri(i128) {
; CHECK-LABEL: divi128ri:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    lea %s2, __divti3@lo
; CHECK-NEXT:    and %s2, %s2, (32)0
; CHECK-NEXT:    lea.sl %s12, __divti3@hi(, %s2)
; CHECK-NEXT:    or %s2, 3, (0)1
; CHECK-NEXT:    or %s3, 0, (0)1
; CHECK-NEXT:    bsic %s10, (, %s12)
; CHECK-NEXT:    or %s11, 0, %s9
  %2 = sdiv i128 %0, 3
  ret i128 %2
}

; Function Attrs: norecurse nounwind readnone
define i64 @divi64ri(i64 %a, i64 %b) {
; CHECK-LABEL: divi64ri:
; CHECK:       # %bb.0:
; CHECK-NEXT:    divs.l %s0, %s0, (62)0
; CHECK-NEXT:    b.l.t (, %s10)
  %r = sdiv i64 %a, 3
  ret i64 %r
}

; Function Attrs: norecurse nounwind readnone
define signext i32 @divi32ri(i32 signext %a, i32 signext %b) {
; CHECK-LABEL: divi32ri:
; CHECK:       # %bb.0:
; CHECK-NEXT:    divs.w.sx %s0, %s0, (62)0
; CHECK-NEXT:    adds.w.sx %s0, %s0, (0)1
; CHECK-NEXT:    b.l.t (, %s10)
  %r = sdiv i32 %a, 3
  ret i32 %r
}

; Function Attrs: norecurse nounwind readnone
define i128 @divu128ri(i128) {
; CHECK-LABEL: divu128ri:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    lea %s2, __udivti3@lo
; CHECK-NEXT:    and %s2, %s2, (32)0
; CHECK-NEXT:    lea.sl %s12, __udivti3@hi(, %s2)
; CHECK-NEXT:    or %s2, 3, (0)1
; CHECK-NEXT:    or %s3, 0, (0)1
; CHECK-NEXT:    bsic %s10, (, %s12)
; CHECK-NEXT:    or %s11, 0, %s9
  %2 = udiv i128 %0, 3
  ret i128 %2
}

; Function Attrs: norecurse nounwind readnone
define i64 @divu64ri(i64 %a, i64 %b) {
; CHECK-LABEL: divu64ri:
; CHECK:       # %bb.0:
; CHECK-NEXT:    divu.l %s0, %s0, (62)0
; CHECK-NEXT:    b.l.t (, %s10)
  %r = udiv i64 %a, 3
  ret i64 %r
}

; Function Attrs: norecurse nounwind readnone
define zeroext i32 @divu32ri(i32 zeroext %a, i32 zeroext %b) {
; CHECK-LABEL: divu32ri:
; CHECK:       # %bb.0:
; CHECK-NEXT:    divu.w %s0, %s0, (62)0
; CHECK-NEXT:    adds.w.zx %s0, %s0, (0)1
; CHECK-NEXT:    b.l.t (, %s10)
  %r = udiv i32 %a, 3
  ret i32 %r
}

; Function Attrs: norecurse nounwind readnone
define i128 @divi128li(i128) {
; CHECK-LABEL: divi128li:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    or %s3, 0, %s1
; CHECK-NEXT:    or %s2, 0, %s0
; CHECK-NEXT:    lea %s0, __divti3@lo
; CHECK-NEXT:    and %s0, %s0, (32)0
; CHECK-NEXT:    lea.sl %s12, __divti3@hi(, %s0)
; CHECK-NEXT:    or %s0, 3, (0)1
; CHECK-NEXT:    or %s1, 0, (0)1
; CHECK-NEXT:    bsic %s10, (, %s12)
; CHECK-NEXT:    or %s11, 0, %s9
  %2 = sdiv i128 3, %0
  ret i128 %2
}

; Function Attrs: norecurse nounwind readnone
define i64 @divi64li(i64 %a, i64 %b) {
; CHECK-LABEL: divi64li:
; CHECK:       # %bb.0:
; CHECK-NEXT:    divs.l %s0, 3, %s1
; CHECK-NEXT:    b.l.t (, %s10)
  %r = sdiv i64 3, %b
  ret i64 %r
}

; Function Attrs: norecurse nounwind readnone
define signext i32 @divi32li(i32 signext %a, i32 signext %b) {
; CHECK-LABEL: divi32li:
; CHECK:       # %bb.0:
; CHECK-NEXT:    divs.w.sx %s0, 3, %s1
; CHECK-NEXT:    adds.w.sx %s0, %s0, (0)1
; CHECK-NEXT:    b.l.t (, %s10)
  %r = sdiv i32 3, %b
  ret i32 %r
}

; Function Attrs: norecurse nounwind readnone
define i128 @divu128li(i128) {
; CHECK-LABEL: divu128li:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    or %s3, 0, %s1
; CHECK-NEXT:    or %s2, 0, %s0
; CHECK-NEXT:    lea %s0, __udivti3@lo
; CHECK-NEXT:    and %s0, %s0, (32)0
; CHECK-NEXT:    lea.sl %s12, __udivti3@hi(, %s0)
; CHECK-NEXT:    or %s0, 3, (0)1
; CHECK-NEXT:    or %s1, 0, (0)1
; CHECK-NEXT:    bsic %s10, (, %s12)
; CHECK-NEXT:    or %s11, 0, %s9
  %2 = udiv i128 3, %0
  ret i128 %2
}

; Function Attrs: norecurse nounwind readnone
define i64 @divu64li(i64 %a, i64 %b) {
; CHECK-LABEL: divu64li:
; CHECK:       # %bb.0:
; CHECK-NEXT:    divu.l %s0, 3, %s1
; CHECK-NEXT:    b.l.t (, %s10)
  %r = udiv i64 3, %b
  ret i64 %r
}

; Function Attrs: norecurse nounwind readnone
define zeroext i32 @divu32li(i32 zeroext %a, i32 zeroext %b) {
; CHECK-LABEL: divu32li:
; CHECK:       # %bb.0:
; CHECK-NEXT:    divu.w %s0, 3, %s1
; CHECK-NEXT:    adds.w.zx %s0, %s0, (0)1
; CHECK-NEXT:    b.l.t (, %s10)
  %r = udiv i32 3, %b
  ret i32 %r
}
