; RUN: llc %s -o - -mtriple=aarch64-unknown -mattr=fuse-fcsel | FileCheck %s
; RUN: llc %s -o - -mtriple=aarch64-unknown -mcpu=apple-a16 | FileCheck %s

; Check that FCMP and FCSEL are adjacent in the output.
; CHECK-LABEL: test_fcmp_fcsels:
; CHECK: fcmp s
; CHECK-NEXT: fcsel s

define float @test_fcmp_fcsels(float %a0, float %a1, float %a2) {
entry:
  %cond = fcmp oeq float %a0, %a1
  %v1 = fadd float %a1, 7.0
  %v2 = select i1 %cond, float %a0, float %v1
  ret float %v2
}

; CHECK-LABEL: test_fcmp_fcseld:
; CHECK: fcmp d
; CHECK-NEXT: fcsel d

define double @test_fcmp_fcseld(double %a0, double %a1, double %a2) {
entry:
  %cond = fcmp oeq double %a0, %a1
  %v1 = fadd double %a1, 7.0
  %v2 = select i1 %cond, double %a0, double %v1
  ret double %v2
}
