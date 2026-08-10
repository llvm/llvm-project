; RUN: llc < %s -mtriple=ve-unknown-unknown | FileCheck %s

; Function Attrs: norecurse nounwind readnone
define signext i8 @f2c(float %a) {
; CHECK-LABEL: f2c:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    cvt.w.s.sx.rz %s0, %s0
; CHECK-NEXT:    adds.w.sx %s0, %s0, (0)1
; CHECK-NEXT:    b.l.t (, %s10)
entry:
  %conv = fptosi float %a to i8
  ret i8 %conv
}

; Function Attrs: norecurse nounwind readnone
define signext i16 @f2s(float %a) {
; CHECK-LABEL: f2s:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    cvt.w.s.sx.rz %s0, %s0
; CHECK-NEXT:    adds.w.sx %s0, %s0, (0)1
; CHECK-NEXT:    b.l.t (, %s10)
entry:
  %conv = fptosi float %a to i16
  ret i16 %conv
}

; Function Attrs: norecurse nounwind readnone
define i32 @f2i(float %a) {
; CHECK-LABEL: f2i:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    cvt.w.s.sx.rz %s0, %s0
; CHECK-NEXT:    b.l.t (, %s10)
entry:
  %conv = fptosi float %a to i32
  ret i32 %conv
}

; Function Attrs: norecurse nounwind readnone
define i64 @f2l(float %a) {
; CHECK-LABEL: f2l:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    cvt.d.s %s0, %s0
; CHECK-NEXT:    cvt.l.d.rz %s0, %s0
; CHECK-NEXT:    b.l.t (, %s10)
entry:
  %conv = fptosi float %a to i64
  ret i64 %conv
}

; Function Attrs: norecurse nounwind readnone
define zeroext i8 @f2uc(float %a) {
; CHECK-LABEL: f2uc:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    cvt.w.s.sx.rz %s0, %s0
; CHECK-NEXT:    adds.w.zx %s0, %s0, (0)1
; CHECK-NEXT:    b.l.t (, %s10)
entry:
  %conv = fptoui float %a to i8
  ret i8 %conv
}

; Function Attrs: norecurse nounwind readnone
define zeroext i16 @f2us(float %a) {
; CHECK-LABEL: f2us:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    cvt.w.s.sx.rz %s0, %s0
; CHECK-NEXT:    adds.w.zx %s0, %s0, (0)1
; CHECK-NEXT:    b.l.t (, %s10)
entry:
  %conv = fptoui float %a to i16
  ret i16 %conv
}

; Function Attrs: norecurse nounwind readnone
define i32 @f2ui(float %a) {
; CHECK-LABEL: f2ui:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    cvt.d.s %s0, %s0
; CHECK-NEXT:    cvt.l.d.rz %s0, %s0
; CHECK-NEXT:    b.l.t (, %s10)
entry:
  %conv = fptoui float %a to i32
  ret i32 %conv
}

; Function Attrs: norecurse nounwind readnone
define i64 @f2ul(float %a) {
; CHECK-LABEL: f2ul:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    lea.sl %s1, 1593835520
; CHECK-NEXT:    fcmp.s %s2, %s0, %s1
; CHECK-NEXT:    fsub.s %s1, %s0, %s1
; CHECK-NEXT:    cvt.d.s %s1, %s1
; CHECK-NEXT:    cvt.l.d.rz %s1, %s1
; CHECK-NEXT:    xor %s1, %s1, (1)1
; CHECK-NEXT:    cvt.d.s %s0, %s0
; CHECK-NEXT:    cvt.l.d.rz %s0, %s0
; CHECK-NEXT:    cmov.s.lt %s1, %s0, %s2
; CHECK-NEXT:    or %s0, 0, %s1
; CHECK-NEXT:    b.l.t (, %s10)
entry:
  %conv = fptoui float %a to i64
  ret i64 %conv
}

; Function Attrs: norecurse nounwind readnone
define signext i8 @d2c(double %a) {
; CHECK-LABEL: d2c:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    cvt.w.d.sx.rz %s0, %s0
; CHECK-NEXT:    adds.w.sx %s0, %s0, (0)1
; CHECK-NEXT:    b.l.t (, %s10)
entry:
  %conv = fptosi double %a to i8
  ret i8 %conv
}

; Function Attrs: norecurse nounwind readnone
define signext i16 @d2s(double %a) {
; CHECK-LABEL: d2s:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    cvt.w.d.sx.rz %s0, %s0
; CHECK-NEXT:    adds.w.sx %s0, %s0, (0)1
; CHECK-NEXT:    b.l.t (, %s10)
entry:
  %conv = fptosi double %a to i16
  ret i16 %conv
}

; Function Attrs: norecurse nounwind readnone
define i32 @d2i(double %a) {
; CHECK-LABEL: d2i:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    cvt.w.d.sx.rz %s0, %s0
; CHECK-NEXT:    b.l.t (, %s10)
entry:
  %conv = fptosi double %a to i32
  ret i32 %conv
}

; Function Attrs: norecurse nounwind readnone
define i64 @d2l(double %a) {
; CHECK-LABEL: d2l:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    cvt.l.d.rz %s0, %s0
; CHECK-NEXT:    b.l.t (, %s10)
entry:
  %conv = fptosi double %a to i64
  ret i64 %conv
}

; Function Attrs: norecurse nounwind readnone
define zeroext i8 @d2uc(double %a) {
; CHECK-LABEL: d2uc:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    cvt.w.d.sx.rz %s0, %s0
; CHECK-NEXT:    adds.w.zx %s0, %s0, (0)1
; CHECK-NEXT:    b.l.t (, %s10)
entry:
  %conv = fptoui double %a to i8
  ret i8 %conv
}

; Function Attrs: norecurse nounwind readnone
define zeroext i16 @d2us(double %a) {
; CHECK-LABEL: d2us:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    cvt.w.d.sx.rz %s0, %s0
; CHECK-NEXT:    adds.w.zx %s0, %s0, (0)1
; CHECK-NEXT:    b.l.t (, %s10)
entry:
  %conv = fptoui double %a to i16
  ret i16 %conv
}

; Function Attrs: norecurse nounwind readnone
define i32 @d2ui(double %a) {
; CHECK-LABEL: d2ui:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    cvt.l.d.rz %s0, %s0
; CHECK-NEXT:    b.l.t (, %s10)
entry:
  %conv = fptoui double %a to i32
  ret i32 %conv
}

; Function Attrs: norecurse nounwind readnone
define i64 @d2ul(double %a) {
; CHECK-LABEL: d2ul:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    lea.sl %s1, 1138753536
; CHECK-NEXT:    fcmp.d %s2, %s0, %s1
; CHECK-NEXT:    fsub.d %s1, %s0, %s1
; CHECK-NEXT:    cvt.l.d.rz %s1, %s1
; CHECK-NEXT:    xor %s1, %s1, (1)1
; CHECK-NEXT:    cvt.l.d.rz %s0, %s0
; CHECK-NEXT:    cmov.d.lt %s1, %s0, %s2
; CHECK-NEXT:    or %s0, 0, %s1
; CHECK-NEXT:    b.l.t (, %s10)
entry:
  %conv = fptoui double %a to i64
  ret i64 %conv
}
