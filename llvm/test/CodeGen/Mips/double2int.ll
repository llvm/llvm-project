; RUN: llc -march=mips -mcpu=mips32 < %s | FileCheck %s
; RUN: llc -march=mips -mcpu=mips32r6 < %s | FileCheck %s

define i32 @f1(double %d) nounwind readnone {
entry:
; CHECK: trunc.w.d $f{{[0-9]+}}, $f12
  %conv = fptosi double %d to i32
  ret i32 %conv
}
