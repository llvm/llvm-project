; RUN: llc < %s -mtriple=ve-unknown-unknown | FileCheck %s

; Function Attrs: norecurse nounwind readnone
define float @c2f(i8 signext %a) {
; CHECK-LABEL: c2f:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    cvt.s.w %s0, %s0
; CHECK-NEXT:    b.l.t (, %s10)
entry:
  %conv = sitofp i8 %a to float
  ret float %conv
}

; Function Attrs: norecurse nounwind readnone
define float @s2f(i16 signext %a) {
; CHECK-LABEL: s2f:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    cvt.s.w %s0, %s0
; CHECK-NEXT:    b.l.t (, %s10)
entry:
  %conv = sitofp i16 %a to float
  ret float %conv
}

; Function Attrs: norecurse nounwind readnone
define float @i2f(i32 %a) {
; CHECK-LABEL: i2f:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    cvt.s.w %s0, %s0
; CHECK-NEXT:    b.l.t (, %s10)
entry:
  %conv = sitofp i32 %a to float
  ret float %conv
}

; Function Attrs: norecurse nounwind readnone
define float @l2f(i64 %a) {
; CHECK-LABEL: l2f:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    cvt.d.l %s0, %s0
; CHECK-NEXT:    cvt.s.d %s0, %s0
; CHECK-NEXT:    b.l.t (, %s10)
entry:
  %conv = sitofp i64 %a to float
  ret float %conv
}

; Function Attrs: norecurse nounwind readnone
define float @uc2f(i8 zeroext %a) {
; CHECK-LABEL: uc2f:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    cvt.s.w %s0, %s0
; CHECK-NEXT:    b.l.t (, %s10)
entry:
  %conv = uitofp i8 %a to float
  ret float %conv
}

; Function Attrs: norecurse nounwind readnone
define float @us2f(i16 zeroext %a) {
; CHECK-LABEL: us2f:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    cvt.s.w %s0, %s0
; CHECK-NEXT:    b.l.t (, %s10)
entry:
  %conv = uitofp i16 %a to float
  ret float %conv
}

; Function Attrs: norecurse nounwind readnone
define float @ui2f(i32 %a) {
; CHECK-LABEL: ui2f:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    and %s0, %s0, (32)0
; CHECK-NEXT:    cvt.d.l %s0, %s0
; CHECK-NEXT:    cvt.s.d %s0, %s0
; CHECK-NEXT:    b.l.t (, %s10)
entry:
  %conv = uitofp i32 %a to float
  ret float %conv
}

; Function Attrs: norecurse nounwind readnone
define float @ul2f(i64 %a) {
; CHECK-LABEL: ul2f:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    cmps.l %s2, %s0, (0)1
; CHECK-NEXT:    cvt.d.l %s1, %s0
; CHECK-NEXT:    cvt.s.d %s1, %s1
; CHECK-NEXT:    srl %s3, %s0, 1
; CHECK-NEXT:    and %s0, 1, %s0
; CHECK-NEXT:    or %s0, %s0, %s3
; CHECK-NEXT:    cvt.d.l %s0, %s0
; CHECK-NEXT:    cvt.s.d %s0, %s0
; CHECK-NEXT:    fadd.s %s0, %s0, %s0
; CHECK-NEXT:    cmov.l.lt %s1, %s0, %s2
; CHECK-NEXT:    or %s0, 0, %s1
; CHECK-NEXT:    b.l.t (, %s10)
entry:
  %conv = uitofp i64 %a to float
  ret float %conv
}

; Function Attrs: norecurse nounwind readnone
define double @c2d(i8 signext %a) {
; CHECK-LABEL: c2d:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    cvt.d.w %s0, %s0
; CHECK-NEXT:    b.l.t (, %s10)
entry:
  %conv = sitofp i8 %a to double
  ret double %conv
}

; Function Attrs: norecurse nounwind readnone
define double @s2d(i16 signext %a) {
; CHECK-LABEL: s2d:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    cvt.d.w %s0, %s0
; CHECK-NEXT:    b.l.t (, %s10)
entry:
  %conv = sitofp i16 %a to double
  ret double %conv
}

; Function Attrs: norecurse nounwind readnone
define double @i2d(i32 %a) {
; CHECK-LABEL: i2d:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    cvt.d.w %s0, %s0
; CHECK-NEXT:    b.l.t (, %s10)
entry:
  %conv = sitofp i32 %a to double
  ret double %conv
}

; Function Attrs: norecurse nounwind readnone
define double @l2d(i64 %a) {
; CHECK-LABEL: l2d:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    cvt.d.l %s0, %s0
; CHECK-NEXT:    b.l.t (, %s10)
entry:
  %conv = sitofp i64 %a to double
  ret double %conv
}

; Function Attrs: norecurse nounwind readnone
define double @uc2d(i8 zeroext %a) {
; CHECK-LABEL: uc2d:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    cvt.d.w %s0, %s0
; CHECK-NEXT:    b.l.t (, %s10)
entry:
  %conv = uitofp i8 %a to double
  ret double %conv
}

; Function Attrs: norecurse nounwind readnone
define double @us2d(i16 zeroext %a) {
; CHECK-LABEL: us2d:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    cvt.d.w %s0, %s0
; CHECK-NEXT:    b.l.t (, %s10)
entry:
  %conv = uitofp i16 %a to double
  ret double %conv
}

; Function Attrs: norecurse nounwind readnone
define double @ui2d(i32 %a) {
; CHECK-LABEL: ui2d:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    and %s0, %s0, (32)0
; CHECK-NEXT:    cvt.d.l %s0, %s0
; CHECK-NEXT:    b.l.t (, %s10)
entry:
  %conv = uitofp i32 %a to double
  ret double %conv
}

; Function Attrs: norecurse nounwind readnone
define double @ul2d(i64 %a) {
; CHECK-LABEL: ul2d:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    srl %s1, %s0, 32
; CHECK-NEXT:    lea.sl %s2, 1160773632
; CHECK-NEXT:    or %s1, %s1, %s2
; CHECK-NEXT:    lea %s2, 1048576
; CHECK-NEXT:    lea.sl %s2, -986710016(, %s2)
; CHECK-NEXT:    fadd.d %s1, %s1, %s2
; CHECK-NEXT:    and %s0, %s0, (32)0
; CHECK-NEXT:    lea.sl %s2, 1127219200
; CHECK-NEXT:    or %s0, %s0, %s2
; CHECK-NEXT:    fadd.d %s0, %s0, %s1
; CHECK-NEXT:    b.l.t (, %s10)
entry:
  %conv = uitofp i64 %a to double
  ret double %conv
}
