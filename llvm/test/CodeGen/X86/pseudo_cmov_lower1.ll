; RUN: llc < %s -mtriple=i386-linux-gnu -mattr=+sse2 -o - | FileCheck %s 
; RUN: llc < %s -mtriple=x86_64-linux-gnu -o - | FileCheck %s 

; This test checks that only a single jae gets generated in the final code
; for lowering the CMOV pseudos that get created for this IR.
; CHECK-LABEL: foo1:
; CHECK: jae
; CHECK-NOT: jae
define double @foo1(float %p1, double %p2, double %p3) nounwind {
entry:
  %c1 = fcmp oge float %p1, 0.000000e+00
  %d0 = fadd double %p2, 1.25e0
  %d1 = fadd double %p3, 1.25e0
  %d2 = select i1 %c1, double %d0, double %d1
  %d3 = select i1 %c1, double %d0, double %p2
  %d4 = select i1 %c1, double %p3, double %d1
  %d5 = fsub double %d2, %d3
  %d6 = fadd double %d5, %d4
  ret double %d6
}

; This test checks that only a single jae gets generated in the final code
; for lowering the CMOV pseudos that get created for this IR.
; CHECK-LABEL: foo2:
; CHECK: jae
; CHECK-NOT: jae
define float @foo2(float %p1, float %p2, float %p3) nounwind {
entry:
  %c1 = fcmp oge float %p1, 0.000000e+00
  %d0 = fadd float %p2, 1.25e0
  %d1 = fadd float %p3, 1.25e0
  %d2 = select i1 %c1, float %d0, float %d1
  %d3 = select i1 %c1, float %d1, float %p2
  %d4 = select i1 %c1, float %d0, float %p3
  %d5 = fsub float %d2, %d3
  %d6 = fadd float %d5, %d4
  ret float %d6
}

