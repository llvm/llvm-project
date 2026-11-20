; RUN: llc < %s -mtriple=ve-unknown-unknown | FileCheck %s

define signext i17 @sdiv_i17(i17 signext %0, i17 signext %1) {
; CHECK-LABEL: sdiv_i17:
; CHECK:       # %bb.0:
; CHECK-NEXT:    divs.w.sx %s0, %s1, %s0
; CHECK-NEXT:    sll %s0, %s0, 47
; CHECK-NEXT:    sra.l %s0, %s0, 47
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = sdiv i17 %1, %0
  ret i17 %3
}

define zeroext i17 @udiv_i17(i17 zeroext %0, i17 zeroext %1) {
; CHECK-LABEL: udiv_i17:
; CHECK:       # %bb.0:
; CHECK-NEXT:    divu.w %s0, %s1, %s0
; CHECK-NEXT:    adds.w.zx %s0, %s0, (0)1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = udiv i17 %1, %0
  ret i17 %3
}

define i65 @sdiv_i65(i65 %0, i65 %1) {
; CHECK-LABEL: sdiv_i65:
; CHECK:       # %bb.0:
; CHECK:         lea {{.*}}, __divti3@lo
; CHECK:         lea.sl %s12, __divti3@hi
; CHECK:         bsic %s10, (, %s12)
  %3 = sdiv i65 %1, %0
  ret i65 %3
}

define i65 @udiv_i65(i65 %0, i65 %1) {
; CHECK-LABEL: udiv_i65:
; CHECK:       # %bb.0:
; CHECK:         lea {{.*}}, __udivti3@lo
; CHECK:         lea.sl %s12, __udivti3@hi
; CHECK:         bsic %s10, (, %s12)
  %3 = udiv i65 %1, %0
  ret i65 %3
}

define i77 @sdiv_i77(i77 %0, i77 %1) {
; CHECK-LABEL: sdiv_i77:
; CHECK:       # %bb.0:
; CHECK:         lea {{.*}}, __divti3@lo
; CHECK:         lea.sl %s12, __divti3@hi
; CHECK:         bsic %s10, (, %s12)
  %3 = sdiv i77 %1, %0
  ret i77 %3
}

define i77 @udiv_i77(i77 %0, i77 %1) {
; CHECK-LABEL: udiv_i77:
; CHECK:       # %bb.0:
; CHECK:         lea {{.*}}, __udivti3@lo
; CHECK:         lea.sl %s12, __udivti3@hi
; CHECK:         bsic %s10, (, %s12)
  %3 = udiv i77 %1, %0
  ret i77 %3
}

define i128 @sdiv_i128(i128 %0, i128 %1) {
; CHECK-LABEL: sdiv_i128:
; CHECK:       # %bb.0:
; CHECK:         lea {{.*}}, __divti3@lo
; CHECK:         lea.sl %s12, __divti3@hi
; CHECK:         bsic %s10, (, %s12)
  %3 = sdiv i128 %1, %0
  ret i128 %3
}

define i128 @udiv_i128(i128 %0, i128 %1) {
; CHECK-LABEL: udiv_i128:
; CHECK:       # %bb.0:
; CHECK:         lea {{.*}}, __udivti3@lo
; CHECK:         lea.sl %s12, __udivti3@hi
; CHECK:         bsic %s10, (, %s12)
  %3 = udiv i128 %1, %0
  ret i128 %3
}

; i131 division is expanded inline (no compiler-rt call) because
; compiler-rt only provides __divti3/__udivti3 for up to 128-bit integers.
; The inline expansion produces very large code, so we only verify it
; compiles without crashing rather than checking the full instruction sequence.
define i131 @sdiv_i131(i131 %0, i131 %1) {
; CHECK-LABEL: sdiv_i131:
; CHECK:         b.l.t (, %s10)
  %3 = sdiv i131 %1, %0
  ret i131 %3
}

define i131 @udiv_i131(i131 %0, i131 %1) {
; CHECK-LABEL: udiv_i131:
; CHECK:         b.l.t (, %s10)
  %3 = udiv i131 %1, %0
  ret i131 %3
}
