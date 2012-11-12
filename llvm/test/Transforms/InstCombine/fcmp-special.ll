; RUN: opt < %s -instcombine -S | FileCheck %s

; Infinity

; CHECK: inf0
; CHECK: ret i1 false
define i1 @inf0(double %arg) nounwind readnone {
  %tmp = fcmp ogt double %arg, 0x7FF0000000000000
  ret i1 %tmp
}

; CHECK: inf1
; CHECK: ret i1 true
define i1 @inf1(double %arg) nounwind readnone {
  %tmp = fcmp ule double %arg, 0x7FF0000000000000
  ret i1 %tmp
}

; Negative infinity

; CHECK: ninf0
; CHECK: ret i1 false
define i1 @ninf0(double %arg) nounwind readnone {
  %tmp = fcmp olt double %arg, 0xFFF0000000000000
  ret i1 %tmp
}

; CHECK: ninf1
; CHECK: ret i1 true
define i1 @ninf1(double %arg) nounwind readnone {
  %tmp = fcmp uge double %arg, 0xFFF0000000000000
  ret i1 %tmp
}

; NaNs

; CHECK: nan0
; CHECK: ret i1 false
define i1 @nan0(double %arg) nounwind readnone {
  %tmp = fcmp ord double %arg, 0x7FF00000FFFFFFFF
  ret i1 %tmp
}

; CHECK: nan1
; CHECK: ret i1 false
define i1 @nan1(double %arg) nounwind readnone {
  %tmp = fcmp oeq double %arg, 0x7FF00000FFFFFFFF
  ret i1 %tmp
}

; CHECK: nan2
; CHECK: ret i1 false
define i1 @nan2(double %arg) nounwind readnone {
  %tmp = fcmp olt double %arg, 0x7FF00000FFFFFFFF
  ret i1 %tmp
}

; CHECK: nan3
; CHECK: ret i1 true
define i1 @nan3(double %arg) nounwind readnone {
  %tmp = fcmp uno double %arg, 0x7FF00000FFFFFFFF
  ret i1 %tmp
}

; CHECK: nan4
; CHECK: ret i1 true
define i1 @nan4(double %arg) nounwind readnone {
  %tmp = fcmp une double %arg, 0x7FF00000FFFFFFFF
  ret i1 %tmp
}

; CHECK: nan5
; CHECK: ret i1 true
define i1 @nan5(double %arg) nounwind readnone {
  %tmp = fcmp ult double %arg, 0x7FF00000FFFFFFFF
  ret i1 %tmp
}

; Negative NaN.

; CHECK: nnan0
; CHECK: ret i1 false
define i1 @nnan0(double %arg) nounwind readnone {
  %tmp = fcmp ord double %arg, 0xFFF00000FFFFFFFF
  ret i1 %tmp
}

; CHECK: nnan1
; CHECK: ret i1 false
define i1 @nnan1(double %arg) nounwind readnone {
  %tmp = fcmp oeq double %arg, 0xFFF00000FFFFFFFF
  ret i1 %tmp
}

; CHECK: nnan2
; CHECK: ret i1 false
define i1 @nnan2(double %arg) nounwind readnone {
  %tmp = fcmp olt double %arg, 0xFFF00000FFFFFFFF
  ret i1 %tmp
}

; CHECK: nnan3
; CHECK: ret i1 true
define i1 @nnan3(double %arg) nounwind readnone {
  %tmp = fcmp uno double %arg, 0xFFF00000FFFFFFFF
  ret i1 %tmp
}

; CHECK: nnan4
; CHECK: ret i1 true
define i1 @nnan4(double %arg) nounwind readnone {
  %tmp = fcmp une double %arg, 0xFFF00000FFFFFFFF
  ret i1 %tmp
}

; CHECK: nnan5
; CHECK: ret i1 true
define i1 @nnan5(double %arg) nounwind readnone {
  %tmp = fcmp ult double %arg, 0xFFF00000FFFFFFFF
  ret i1 %tmp
}

; Negative zero.

; CHECK: nzero0
; CHECK: ret i1 true
define i1 @nzero0() {
  %tmp = fcmp oeq double 0.0, -0.0
  ret i1 %tmp
}

; CHECK: nzero1
; CHECK: ret i1 false
define i1 @nzero1() {
  %tmp = fcmp ogt double 0.0, -0.0
  ret i1 %tmp
}

; Misc.

; CHECK: misc0
; CHECK: %tmp = fcmp ord double %arg, 0.000000e+00
; CHECK: ret i1 %tmp
define i1 @misc0(double %arg) {
  %tmp = fcmp oeq double %arg, %arg
  ret i1 %tmp
}

; CHECK: misc1
; CHECK: ret i1 false
define i1 @misc1(double %arg) {
  %tmp = fcmp one double %arg, %arg
  ret i1 %tmp
}

