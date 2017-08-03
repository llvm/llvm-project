; RUN: llc < %s -march=x86 -mcpu=i386 | FileCheck %s
; PR6679

define float @foo(float* %col.2.0) {
; CHECK: fucomp
; CHECK-NOT: fucompi
; CHECK: j
; CHECK-NOT: fcmov
  %tmp = load float, float* %col.2.0
  %tmp16 = fcmp olt float %tmp, 0.000000e+00
  %tmp20 = fsub float -0.000000e+00, %tmp
  %iftmp.2.0 = select i1 %tmp16, float %tmp20, float %tmp
  ret float %iftmp.2.0
}
