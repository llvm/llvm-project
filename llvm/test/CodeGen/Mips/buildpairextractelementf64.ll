; RUN: llc  < %s -march=mipsel | FileCheck %s
; RUN: llc  < %s -march=mips   | FileCheck %s
@a = external global i32

define double @f(i32 %a1, double %d) nounwind {
entry:
; CHECK: mtc1
; CHECK: mtc1
  store i32 %a1, i32* @a, align 4
  %add = fadd double %d, 2.000000e+00
  ret double %add
}

define void @f3(double %d, i32 %a1) nounwind {
entry:
; CHECK: mfc1
; CHECK: mfc1
  tail call void @f2(i32 %a1, double %d) nounwind
  ret void
}

declare void @f2(i32, double)

