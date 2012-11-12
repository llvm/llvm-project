; RUN: llc  < %s -march=mipsel -mcpu=mips32 | FileCheck %s 

define float @foo0(i32 %a, float %d) nounwind readnone {
entry:
; CHECK-NOT: neg.s
  %sub = fsub float -0.000000e+00, %d
  ret float %sub
}

define double @foo1(i32 %a, double %d) nounwind readnone {
entry:
; CHECK:     foo1
; CHECK-NOT: neg.d
; CHECK:     jr
  %sub = fsub double -0.000000e+00, %d
  ret double %sub
}
