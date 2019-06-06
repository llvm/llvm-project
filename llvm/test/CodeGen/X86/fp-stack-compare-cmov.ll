; RUN: llc < %s -mtriple=i686-- -mcpu=pentiumpro | FileCheck %s
; PR1012

define float @foo(float* %col.2.0) {
; CHECK: fucompi
; CHECK: fcmov
  %tmp = load float, float* %col.2.0
  %tmp16 = fcmp olt float %tmp, 0.000000e+00
  %tmp20 = fsub float -0.000000e+00, %tmp
  %iftmp.2.0 = select i1 %tmp16, float %tmp20, float %tmp
  ret float %iftmp.2.0
}

define float @foo_unary_fneg(float* %col.2.0) {
; CHECK: fucompi
; CHECK: fcmov
  %tmp = load float, float* %col.2.0
  %tmp16 = fcmp olt float %tmp, 0.000000e+00
  %tmp20 = fneg float %tmp
  %iftmp.2.0 = select i1 %tmp16, float %tmp20, float %tmp
  ret float %iftmp.2.0
}
