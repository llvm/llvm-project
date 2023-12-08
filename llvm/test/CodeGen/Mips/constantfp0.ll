; RUN: llc -march=mips < %s | FileCheck %s

define i32 @branch(double %d) nounwind readnone {
entry:
; CHECK: mtc1  $zero, $f[[R0:[0-9]+]]
; CHECK: c.eq.d  $f{{[0-9]+}}, $f[[R0]]

  %tobool = fcmp une double %d, 0.000000e+00
  %. = zext i1 %tobool to i32
  ret i32 %.
}
