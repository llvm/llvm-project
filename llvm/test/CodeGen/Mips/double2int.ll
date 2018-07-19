; RUN: llc -march=mips < %s | FileCheck %s

define i32 @f1(double %d) nounwind readnone {
entry:
; CHECK: trunc.w.d $f{{[0-9]+}}, $f12
  %conv = fptosi double %d to i32
  ret i32 %conv
}
