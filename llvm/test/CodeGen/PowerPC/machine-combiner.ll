; RUN: llc -verify-machineinstrs -O3 -mcpu=pwr7 < %s | FileCheck  %s -check-prefix=CHECK -check-prefix=CHECK-PWR
; RUN: llc -verify-machineinstrs -O3 -mcpu=a2q < %s | FileCheck  %s -check-prefix=CHECK -check-prefix=CHECK-QPX
; RUN: llc -verify-machineinstrs -O3 -mcpu=pwr9 < %s | FileCheck  %s -check-prefix=FIXPOINT
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

  %t0 = fadd reassoc nsz float %x0, %x1
  %t1 = fadd reassoc nsz float %t0, %x2
  %t2 = fadd reassoc nsz float %t1, %x3
  ret float %t2
}

define float @reassociate_adds2(float %x0, float %x1, float %x2, float %x3) {
; CHECK-LABEL: reassociate_adds2:
; CHECK:       # %bb.0:
; CHECK:       fadds [[REG0:[0-9]+]], 1, 2
; CHECK:       fadds [[REG1:[0-9]+]], 3, 4
; CHECK:       fadds 1, [[REG0]], [[REG1]]
; CHECK-NEXT:  blr

  %t0 = fadd reassoc nsz float %x0, %x1
  %t1 = fadd reassoc nsz float %x2, %t0
  %t2 = fadd reassoc nsz float %t1, %x3
  ret float %t2
}

define float @reassociate_adds3(float %x0, float %x1, float %x2, float %x3) {
; CHECK-LABEL: reassociate_adds3:
; CHECK:       # %bb.0:
; CHECK:       fadds [[REG0:[0-9]+]], 1, 2
; CHECK:       fadds [[REG1:[0-9]+]], 3, 4
; CHECK:       fadds 1, [[REG0]], [[REG1]]
; CHECK-NEXT:  blr

  %t0 = fadd reassoc nsz float %x0, %x1
  %t1 = fadd reassoc nsz float %t0, %x2
  %t2 = fadd reassoc nsz float %x3, %t1
  ret float %t2
}

define float @reassociate_adds4(float %x0, float %x1, float %x2, float %x3) {
; CHECK-LABEL: reassociate_adds4:
; CHECK:       # %bb.0:
; CHECK:       fadds [[REG0:[0-9]+]], 1, 2
; CHECK:       fadds [[REG1:[0-9]+]], 3, 4
; CHECK:       fadds 1, [[REG0]], [[REG1]]
; CHECK-NEXT:  blr

  %t0 = fadd reassoc nsz float %x0, %x1
  %t1 = fadd reassoc nsz float %x2, %t0
  %t2 = fadd reassoc nsz float %x3, %t1
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
; CHECK-DAG:   fadds [[REG13:[0-9]+]], [[REG12]], 7
; CHECK-DAG:   fadds [[REG1:[0-9]+]], [[REG0]], [[REG11]]
; CHECK-DAG:   fadds [[REG2:[0-9]+]], [[REG1]], [[REG13]]
; CHECK:       fadds 1, [[REG2]], 8
; CHECK-NEXT:    blr

  %t0 = fadd reassoc nsz float %x0, %x1
  %t1 = fadd reassoc nsz float %t0, %x2
  %t2 = fadd reassoc nsz float %t1, %x3
  %t3 = fadd reassoc nsz float %t2, %x4
  %t4 = fadd reassoc nsz float %t3, %x5
  %t5 = fadd reassoc nsz float %t4, %x6
  %t6 = fadd reassoc nsz float %t5, %x7
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

  %t0 = fadd reassoc nsz <4 x float> %x0, %x1
  %t1 = fadd reassoc nsz <4 x float> %t0, %x2
  %t2 = fadd reassoc nsz <4 x float> %t1, %x3
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

  %t0 = fadd reassoc nsz <4 x float> %x0, %x1
  %t1 = fadd reassoc nsz <4 x float> %x2, %t0
  %t2 = fadd reassoc nsz <4 x float> %t1, %x3
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

  %t0 = fadd reassoc nsz <4 x float> %x0, %x1
  %t1 = fadd reassoc nsz <4 x float> %t0, %x2
  %t2 = fadd reassoc nsz <4 x float> %x3, %t1
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

  %t0 = fadd reassoc nsz <4 x float> %x0, %x1
  %t1 = fadd reassoc nsz <4 x float> %x2, %t0
  %t2 = fadd reassoc nsz <4 x float> %x3, %t1
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

define i32 @reassociate_mullw(i32 %x0, i32 %x1, i32 %x2, i32 %x3) {
; FIXPOINT-LABEL: reassociate_mullw:
; FIXPOINT:       # %bb.0:
; FIXPOINT:       mullw [[REG0:[0-9]+]], 3, 4
; FIXPOINT:       mullw [[REG1:[0-9]+]], 5, 6
; FIXPOINT:       mullw 3, [[REG0]], [[REG1]]
; FIXPOINT-NEXT:  blr

  %t0 = mul i32 %x0, %x1
  %t1 = mul i32 %t0, %x2
  %t2 = mul i32 %t1, %x3
  ret i32 %t2
}

define i64 @reassociate_mulld(i64 %x0, i64 %x1, i64 %x2, i64 %x3) {
; FIXPOINT-LABEL: reassociate_mulld:
; FIXPOINT:       # %bb.0:
; FIXPOINT:       mulld [[REG0:[0-9]+]], 3, 4
; FIXPOINT:       mulld [[REG1:[0-9]+]], 5, 6
; FIXPOINT:       mulld 3, [[REG0]], [[REG1]]
; FIXPOINT-NEXT:  blr

  %t0 = mul i64 %x0, %x1
  %t1 = mul i64 %t0, %x2
  %t2 = mul i64 %t1, %x3
  ret i64 %t2
}

define double @reassociate_mamaa_double(double %0, double %1, double %2, double %3, double %4, double %5) {
; CHECK-LABEL: reassociate_mamaa_double:
; CHECK:       # %bb.0:
; CHECK-QPX-DAG:   fmadd [[REG0:[0-9]+]], 4, 3, 2
; CHECK-QPX-DAG:   fmadd [[REG1:[0-9]+]], 6, 5, 1
; CHECK-QPX:       fadd 1, [[REG0]], [[REG1]]
; CHECK-PWR-DAG:   xsmaddadp 1, 6, 5
; CHECK-PWR-DAG:   xsmaddadp 2, 4, 3
; CHECK-PWR:       xsadddp 1, 2, 1
; CHECK-NEXT:  blr
  %7 = fmul reassoc nsz double %3, %2
  %8 = fmul reassoc nsz double %5, %4
  %9 = fadd reassoc nsz double %1, %0
  %10 = fadd reassoc nsz double %9, %7
  %11 = fadd reassoc nsz double %10, %8
  ret double %11
}

define float @reassociate_mamaa_float(float %0, float %1, float %2, float %3, float %4, float %5) {
; CHECK-LABEL: reassociate_mamaa_float:
; CHECK:       # %bb.0:
; CHECK-DAG:   fmadds [[REG0:[0-9]+]], 4, 3, 2
; CHECK-DAG:   fmadds [[REG1:[0-9]+]], 6, 5, 1
; CHECK:       fadds 1, [[REG0]], [[REG1]]
; CHECK-NEXT:  blr
  %7 = fmul reassoc nsz float %3, %2
  %8 = fmul reassoc nsz float %5, %4
  %9 = fadd reassoc nsz float %1, %0
  %10 = fadd reassoc nsz float %9, %7
  %11 = fadd reassoc nsz float %10, %8
  ret float %11
}

define <4 x float> @reassociate_mamaa_vec(<4 x float> %0, <4 x float> %1, <4 x float> %2, <4 x float> %3, <4 x float> %4, <4 x float> %5) {
; CHECK-LABEL: reassociate_mamaa_vec:
; CHECK:       # %bb.0:
; CHECK-QPX-DAG:   qvfmadds [[REG0:[0-9]+]], 4, 3, 2
; CHECK-QPX-DAG:   qvfmadds [[REG1:[0-9]+]], 6, 5, 1
; CHECK-QPX:       qvfadds 1, [[REG0]], [[REG1]]
; CHECK-PWR-DAG:   xvmaddasp [[REG0:[0-9]+]], 39, 38
; CHECK-PWR-DAG:   xvmaddasp [[REG1:[0-9]+]], 37, 36
; CHECK-PWR:       xvaddsp 34, [[REG1]], [[REG0]]
; CHECK-NEXT:  blr
  %7 = fmul reassoc nsz <4 x float> %3, %2
  %8 = fmul reassoc nsz <4 x float> %5, %4
  %9 = fadd reassoc nsz <4 x float> %1, %0
  %10 = fadd reassoc nsz <4 x float> %9, %7
  %11 = fadd reassoc nsz <4 x float> %10, %8
  ret <4 x float> %11
}

define double @reassociate_mamama_double(double %0, double %1, double %2, double %3, double %4, double %5, double %6, double %7, double %8) {
; CHECK-LABEL: reassociate_mamama_double:
; CHECK:       # %bb.0:
; CHECK-QPX:       fmadd [[REG0:[0-9]+]], 2, 1, 7
; CHECK-QPX-DAG:   fmul [[REG1:[0-9]+]], 4, 3
; CHECK-QPX-DAG:   fmadd [[REG2:[0-9]+]], 6, 5, [[REG0]]
; CHECK-QPX-DAG:   fmadd [[REG3:[0-9]+]], 9, 8, [[REG1]]
; CHECK-QPX:       fadd 1, [[REG2]], [[REG3]]
; CHECK-PWR:       xsmaddadp 7, 2, 1
; CHECK-PWR-DAG:   xsmuldp [[REG0:[0-9]+]], 4, 3
; CHECK-PWR-DAG:   xsmaddadp 7, 6, 5
; CHECK-PWR-DAG:   xsmaddadp [[REG0]], 9, 8
; CHECK-PWR:       xsadddp 1, 7, [[REG0]]
; CHECK-NEXT:  blr
  %10 = fmul reassoc nsz double %1, %0
  %11 = fmul reassoc nsz double %3, %2
  %12 = fmul reassoc nsz double %5, %4
  %13 = fmul reassoc nsz double %8, %7
  %14 = fadd reassoc nsz double %11, %10
  %15 = fadd reassoc nsz double %14, %6
  %16 = fadd reassoc nsz double %15, %12
  %17 = fadd reassoc nsz double %16, %13
  ret double %17
}

define dso_local float @reassociate_mamama_8(float %0, float %1, float %2, float %3, float %4, float %5, float %6, float %7, float %8,
                                             float %9, float %10, float %11, float %12, float %13, float %14, float %15, float %16) {
; CHECK-LABEL: reassociate_mamama_8:
; CHECK:       # %bb.0:
; CHECK-DAG:    fmadds [[REG0:[0-9]+]], 3, 2, 1
; CHECK-DAG:    fmuls  [[REG1:[0-9]+]], 5, 4
; CHECK-DAG:    fmadds [[REG2:[0-9]+]], 7, 6, [[REG0]]
; CHECK-DAG:    fmadds [[REG3:[0-9]+]], 9, 8, [[REG1]]
;
; CHECK-DAG:    fmadds [[REG4:[0-9]+]], 13, 12, [[REG3]]
; CHECK-DAG:    fmadds [[REG5:[0-9]+]], 11, 10, [[REG2]]
;
; CHECK-DAG:    fmadds [[REG6:[0-9]+]], 3, 2, [[REG4]]
; CHECK-DAG:    fmadds [[REG7:[0-9]+]], 5, 4, [[REG5]]
; CHECK:        fadds 1, [[REG7]], [[REG6]]
; CHECK-NEXT:   blr
  %18 = fmul reassoc nsz float %2, %1
  %19 = fadd reassoc nsz float %18, %0
  %20 = fmul reassoc nsz float %4, %3
  %21 = fadd reassoc nsz float %19, %20
  %22 = fmul reassoc nsz float %6, %5
  %23 = fadd reassoc nsz float %21, %22
  %24 = fmul reassoc nsz float %8, %7
  %25 = fadd reassoc nsz float %23, %24
  %26 = fmul reassoc nsz float %10, %9
  %27 = fadd reassoc nsz float %25, %26
  %28 = fmul reassoc nsz float %12, %11
  %29 = fadd reassoc nsz float %27, %28
  %30 = fmul reassoc nsz float %14, %13
  %31 = fadd reassoc nsz float %29, %30
  %32 = fmul reassoc nsz float %16, %15
  %33 = fadd reassoc nsz float %31, %32
  ret float %33
}

