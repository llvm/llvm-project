; RUN: llc -march=mips -mfix4300 -verify-machineinstrs < %s | FileCheck %s

; Function Attrs: mustprogress nofree norecurse nosync nounwind readnone willreturn
define dso_local float @fun_s(float %x) local_unnamed_addr #0 {
entry:
; CHECK-LABEL: fun_s
; CHECK: mul.s
; CHECK-NEXT: nop
; CHECK: mul.s
  %mul = fmul float %x, %x
  %mul1 = fmul float %mul, %x
  ret float %mul1
}

define dso_local double @fun_d(double %x) local_unnamed_addr #0 {
entry:
; CHECK-LABEL: fun_d
; CHECK: mul.d
; CHECK-NEXT: nop
; CHECK: mul.d
  %mul = fmul double %x, %x
  %mul1 = fmul double %mul, %x
  ret double %mul1
}
