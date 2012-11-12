; RUN: opt < %s -instcombine -S | FileCheck %s

; x != y ? x : y -> x if it's the right kind of != and at least
; one of x and y is not negative zero.

; CHECK: f0
; CHECK: ret double %x
define double @f0(double %x) nounwind readnone {
entry:
  %cmp = fcmp une double %x, -1.0
  %cond = select i1 %cmp, double %x, double -1.0
  ret double %cond
}
; CHECK: f1
; CHECK: ret double -1.000000e+00
define double @f1(double %x) nounwind readnone {
entry:
  %cmp = fcmp une double %x, -1.0
  %cond = select i1 %cmp, double -1.0, double %x
  ret double %cond
}
; CHECK: f2
; CHECK: ret double %cond
define double @f2(double %x, double %y) nounwind readnone {
entry:
  %cmp = fcmp une double %x, %y
  %cond = select i1 %cmp, double %x, double %y
  ret double %cond
}
; CHECK: f3
; CHECK: ret double %cond
define double @f3(double %x, double %y) nounwind readnone {
entry:
  %cmp = fcmp une double %x, %y
  %cond = select i1 %cmp, double %y, double %x
  ret double %cond
}
; CHECK: f4
; CHECK: ret double %cond
define double @f4(double %x) nounwind readnone {
entry:
  %cmp = fcmp one double %x, -1.0
  %cond = select i1 %cmp, double %x, double -1.0
  ret double %cond
}
; CHECK: f5
; CHECK: ret double %cond
define double @f5(double %x) nounwind readnone {
entry:
  %cmp = fcmp one double %x, -1.0
  %cond = select i1 %cmp, double -1.0, double %x
  ret double %cond
}
