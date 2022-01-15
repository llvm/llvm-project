; RUN: llc -march=mips -mfix4300 -verify-machineinstrs < %s | FileCheck %s

; Function Attrs: nounwind
define dso_local void @fun_s(float %a) local_unnamed_addr #0 {
entry:
; CHECK-LABEL: fun_s
; CHECK: mul.s
; CHECK-NEXT: nop
  %mul = fmul float %a, %a
  tail call void @foo_s(float %mul) #2
  ret void
}

declare dso_local void @foo_s(float) local_unnamed_addr #1

; Function Attrs: nounwind
define dso_local void @fun_d(double %a) local_unnamed_addr #0 {
entry:
; CHECK-LABEL: fun_d
; CHECK: mul.d
; CHECK-NEXT: nop
  %mul = fmul double %a, %a
  tail call void @foo_d(double %mul) #2
  ret void
}

declare dso_local void @foo_d(double) local_unnamed_addr #1
