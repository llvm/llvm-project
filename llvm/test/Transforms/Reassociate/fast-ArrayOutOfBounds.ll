; RUN: opt < %s -passes=reassociate,instcombine -S | FileCheck %s

; Not marked as fast, so must not change.
define float @test1(float %a0, float %a1, float %a2, float %a3, float %a4) {
; CHECK-LABEL: test1
; CHECK-NEXT: %tmp.2 = fadd float %a3, %a4
; CHECK-NEXT: %tmp.4 = fadd float %tmp.2, %a2
; CHECK-NEXT: %tmp.6 = fadd float %tmp.4, %a1
; CHECK-NEXT: %tmp.8 = fadd float %tmp.6, %a0
; CHECK-NEXT: %tmp.11 = fadd float %a2, %a3
; CHECK-NEXT: %tmp.13 = fadd float %tmp.11, %a1
; CHECK-NEXT: %tmp.15 = fadd float %tmp.13, %a0
; CHECK-NEXT: %tmp.18 = fadd float %a1, %a2
; CHECK-NEXT: %tmp.20 = fadd float %tmp.18, %a0
; CHECK-NEXT: %tmp.23 = fadd float %a0, %a1
; CHECK-NEXT: %tmp.26 = fsub float %tmp.8, %tmp.15
; CHECK-NEXT: %tmp.28 = fadd float %tmp.20, %tmp.26
; CHECK-NEXT: %tmp.30 = fsub float %tmp.28, %tmp.23
; CHECK-NEXT: %tmp.32 = fsub float %tmp.30, %a4
; CHECK-NEXT: %tmp.34 = fsub float %tmp.32, %a2
; CHECK-NEXT: %T = fmul float %tmp.34, %tmp.34
; CHECK-NEXT: ret float %T

  %tmp.2 = fadd float %a4, %a3
  %tmp.4 = fadd float %tmp.2, %a2
  %tmp.6 = fadd float %tmp.4, %a1
  %tmp.8 = fadd float %tmp.6, %a0
  %tmp.11 = fadd float %a3, %a2
  %tmp.13 = fadd float %tmp.11, %a1
  %tmp.15 = fadd float %tmp.13, %a0
  %tmp.18 = fadd float %a2, %a1
  %tmp.20 = fadd float %tmp.18, %a0
  %tmp.23 = fadd float %a1, %a0
  %tmp.26 = fsub float %tmp.8, %tmp.15
  %tmp.28 = fadd float %tmp.26, %tmp.20
  %tmp.30 = fsub float %tmp.28, %tmp.23
  %tmp.32 = fsub float %tmp.30, %a4
  %tmp.34 = fsub float %tmp.32, %a2
  %T = fmul float %tmp.34, %tmp.34
  ret float %T
}

; Should be able to eliminate everything.
define float @test2(float %a0, float %a1, float %a2, float %a3, float %a4) {
; CHECK-LABEL: test2
; CHECK: ret float 0.000000e+00

  %tmp.2 = fadd fast float %a4, %a3
  %tmp.4 = fadd fast float %tmp.2, %a2
  %tmp.6 = fadd fast float %tmp.4, %a1
  %tmp.8 = fadd fast float %tmp.6, %a0
  %tmp.11 = fadd fast float %a3, %a2
  %tmp.13 = fadd fast float %tmp.11, %a1
  %tmp.15 = fadd fast float %tmp.13, %a0
  %tmp.18 = fadd fast float %a2, %a1
  %tmp.20 = fadd fast float %tmp.18, %a0
  %tmp.23 = fadd fast float %a1, %a0
  %tmp.26 = fsub fast float %tmp.8, %tmp.15
  %tmp.28 = fadd fast float %tmp.26, %tmp.20
  %tmp.30 = fsub fast float %tmp.28, %tmp.23
  %tmp.32 = fsub fast float %tmp.30, %a4
  %tmp.34 = fsub fast float %tmp.32, %a2
  %T = fmul fast float %tmp.34, %tmp.34
  ret float %T
}
