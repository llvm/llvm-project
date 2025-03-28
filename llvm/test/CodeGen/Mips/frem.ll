; RUN: llc < %s -mtriple=mipsel 

define float @fmods(float %x, float %y) {
entry:
  %r = frem float %x, %y
  ret float %r
}

define double @fmodd(double %x, double %y) {
entry:
  %r = frem double %x, %y
  ret double %r
}
