; RUN: llc -verify-machineinstrs -O3 -mcpu=pwr7 -enable-unsafe-fp-math < %s | FileCheck -allow-deprecated-dag-overlap %s -check-prefix=CHECK -check-prefix=CHECK-PWR
; RUN: llc -verify-machineinstrs -O3 -mcpu=a2q -enable-unsafe-fp-math < %s | FileCheck -allow-deprecated-dag-overlap %s -check-prefix=CHECK -check-prefix=CHECK-QPX
target datalayout = "E-m:e-i64:64-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

; Verify that the first two adds are independent regardless of how the inputs are
; commuted. The destination registers are used as source registers for the third add.

define float @reassociate_adds1(float %x0, float %x1, float %x2, float %x3) {
; CHECK-LABEL: reassociate_adds1:
; CHECK:       # %bb.0:
; CHECK:       fadds [[REG0:[0-9]+]], 1, 2
; CHECK:       fadds [[REG1:[0-9]+]], 3, 4
; CHECK:       fadds 1, [[REG0]], [[REG1]]
; CHECK-NEXT:  blr

  %t0 = fadd float %x0, %x1
  %t1 = fadd float %t0, %x2
  %t2 = fadd float %t1, %x3
  ret float %t2
}

define float @reassociate_adds2(float %x0, float %x1, float %x2, float %x3) {
; CHECK-LABEL: reassociate_adds2:
; CHECK:       # %bb.0:
; CHECK:       fadds [[REG0:[0-9]+]], 1, 2
; CHECK:       fadds [[REG1:[0-9]+]], 3, 4
; CHECK:       fadds 1, [[REG0]], [[REG1]]
; CHECK-NEXT:  blr

  %t0 = fadd float %x0, %x1
  %t1 = fadd float %x2, %t0
  %t2 = fadd float %t1, %x3
  ret float %t2
}

define float @reassociate_adds3(float %x0, float %x1, float %x2, float %x3) {
; CHECK-LABEL: reassociate_adds3:
; CHECK:       # %bb.0:
; CHECK:       fadds [[REG0:[0-9]+]], 1, 2
; CHECK:       fadds [[REG1:[0-9]+]], 3, 4
; CHECK:       fadds 1, [[REG0]], [[REG1]]
; CHECK-NEXT:  blr

  %t0 = fadd float %x0, %x1
  %t1 = fadd float %t0, %x2
  %t2 = fadd float %x3, %t1
  ret float %t2
}

define float @reassociate_adds4(float %x0, float %x1, float %x2, float %x3) {
; CHECK-LABEL: reassociate_adds4:
; CHECK:       # %bb.0:
; CHECK:       fadds [[REG0:[0-9]+]], 1, 2
; CHECK:       fadds [[REG1:[0-9]+]], 3, 4
; CHECK:       fadds 1, [[REG0]], [[REG1]]
; CHECK-NEXT:  blr

  %t0 = fadd float %x0, %x1
  %t1 = fadd float %x2, %t0
  %t2 = fadd float %x3, %t1
  ret float %t2
}

; Verify that we reassociate some of these ops. The optimal balanced tree of adds is not
; produced because that would cost more compile time.

define float @reassociate_adds5(float %x0, float %x1, float %x2, float %x3, float %x4, float %x5, float %x6, float %x7) {
; CHECK-LABEL: reassociate_adds5:
; CHECK:       # %bb.0:
; CHECK-DAG:   fadds [[REG12:[0-9]+]], 5, 6
; CHECK-DAG:   fadds [[REG0:[0-9]+]], 1, 2
; CHECK-DAG:   fadds [[REG11:[0-9]+]], 3, 4
; CHECK:       fadds [[REG13:[0-9]+]], [[REG12]], 7
; CHECK-DAG:   fadds [[REG1:[0-9]+]], [[REG0]], [[REG11]]
; CHECK-DAG:   fadds [[REG2:[0-9]+]], [[REG1]], [[REG13]]
; CHECK:       fadds 1, [[REG2]], 8
; CHECK-NEXT:    blr

  %t0 = fadd float %x0, %x1
  %t1 = fadd float %t0, %x2
  %t2 = fadd float %t1, %x3
  %t3 = fadd float %t2, %x4
  %t4 = fadd float %t3, %x5
  %t5 = fadd float %t4, %x6
  %t6 = fadd float %t5, %x7
  ret float %t6
}

; Verify that we reassociate vector instructions too.

define <4 x float> @vector_reassociate_adds1(<4 x float> %x0, <4 x float> %x1, <4 x float> %x2, <4 x float> %x3) {
; CHECK-LABEL: vector_reassociate_adds1:
; CHECK:       # %bb.0:
; CHECK-QPX:       qvfadds [[REG0:[0-9]+]], 1, 2
; CHECK-QPX:       qvfadds [[REG1:[0-9]+]], 3, 4
; CHECK-QPX:       qvfadds 1, [[REG0]], [[REG1]]
; CHECK-PWR:       xvaddsp [[REG0:[0-9]+]], 34, 35
; CHECK-PWR:       xvaddsp [[REG1:[0-9]+]], 36, 37
; CHECK-PWR:       xvaddsp 34, [[REG0]], [[REG1]]
; CHECK-NEXT:  blr

  %t0 = fadd <4 x float> %x0, %x1
  %t1 = fadd <4 x float> %t0, %x2
  %t2 = fadd <4 x float> %t1, %x3
  ret <4 x float> %t2
}

define <4 x float> @vector_reassociate_adds2(<4 x float> %x0, <4 x float> %x1, <4 x float> %x2, <4 x float> %x3) {
; CHECK-LABEL: vector_reassociate_adds2:
; CHECK:       # %bb.0:
; CHECK-QPX:       qvfadds [[REG0:[0-9]+]], 1, 2
; CHECK-QPX:       qvfadds [[REG1:[0-9]+]], 3, 4
; CHECK-QPX:       qvfadds 1, [[REG0]], [[REG1]]
; CHECK-PWR:       xvaddsp [[REG0:[0-9]+]], 34, 35
; CHECK-PWR:       xvaddsp [[REG1:[0-9]+]], 36, 37
; CHECK-PWR:       xvaddsp 34, [[REG0]], [[REG1]]
; CHECK-NEXT:  blr

  %t0 = fadd <4 x float> %x0, %x1
  %t1 = fadd <4 x float> %x2, %t0
  %t2 = fadd <4 x float> %t1, %x3
  ret <4 x float> %t2
}

define <4 x float> @vector_reassociate_adds3(<4 x float> %x0, <4 x float> %x1, <4 x float> %x2, <4 x float> %x3) {
; CHECK-LABEL: vector_reassociate_adds3:
; CHECK:       # %bb.0:
; CHECK-QPX:       qvfadds [[REG0:[0-9]+]], 1, 2
; CHECK-QPX:       qvfadds [[REG1:[0-9]+]], 3, 4
; CHECK-QPX:       qvfadds 1, [[REG0]], [[REG1]]
; CHECK-PWR:       xvaddsp [[REG0:[0-9]+]], 34, 35
; CHECK-PWR:       xvaddsp [[REG1:[0-9]+]], 36, 37
; CHECK-PWR:       xvaddsp 34, [[REG0]], [[REG1]]
; CHECK-NEXT:  blr

  %t0 = fadd <4 x float> %x0, %x1
  %t1 = fadd <4 x float> %t0, %x2
  %t2 = fadd <4 x float> %x3, %t1
  ret <4 x float> %t2
}

define <4 x float> @vector_reassociate_adds4(<4 x float> %x0, <4 x float> %x1, <4 x float> %x2, <4 x float> %x3) {
; CHECK-LABEL: vector_reassociate_adds4:
; CHECK:       # %bb.0:
; CHECK-QPX:       qvfadds [[REG0:[0-9]+]], 1, 2
; CHECK-QPX:       qvfadds [[REG1:[0-9]+]], 3, 4
; CHECK-QPX:       qvfadds 1, [[REG0]], [[REG1]]
; CHECK-PWR:       xvaddsp [[REG0:[0-9]+]], 34, 35
; CHECK-PWR:       xvaddsp [[REG1:[0-9]+]], 36, 37
; CHECK-PWR:       xvaddsp 34, [[REG0]], [[REG1]]
; CHECK-NEXT:  blr

  %t0 = fadd <4 x float> %x0, %x1
  %t1 = fadd <4 x float> %x2, %t0
  %t2 = fadd <4 x float> %x3, %t1
  ret <4 x float> %t2
}

define float @reassociate_adds6(float %x0, float %x1, float %x2, float %x3) {
  %t0 = fdiv float %x0, %x1
  %t1 = fadd float %x2, %t0
  %t2 = fadd float %x3, %t1
  ret float %t2
}

define float @reassociate_muls1(float %x0, float %x1, float %x2, float %x3) {
  %t0 = fdiv float %x0, %x1
  %t1 = fmul float %x2, %t0
  %t2 = fmul float %x3, %t1
  ret float %t2
}

define double @reassociate_adds_double(double %x0, double %x1, double %x2, double %x3) {
  %t0 = fdiv double %x0, %x1
  %t1 = fadd double %x2, %t0
  %t2 = fadd double %x3, %t1
  ret double %t2
}

define double @reassociate_muls_double(double %x0, double %x1, double %x2, double %x3) {
  %t0 = fdiv double %x0, %x1
  %t1 = fmul double %x2, %t0
  %t2 = fmul double %x3, %t1
  ret double %t2
}


