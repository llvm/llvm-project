; RUN: llc < %s -mtriple=armv7-apple-ios6 -mcpu=cortex-a8 | FileCheck %s --check-prefix=NOOPT
; RUN: llc < %s -mtriple=armv7-apple-ios7 -mcpu=cortex-a8 | FileCheck %s --check-prefix=SINCOS
; RUN: llc < %s -mtriple=armv7-linux-gnu -mcpu=cortex-a8 | FileCheck %s --check-prefix=SINCOS-GNU
; RUN: llc < %s -mtriple=armv7-linux-gnueabi -mcpu=cortex-a8 \
; RUN:   --enable-unsafe-fp-math | FileCheck %s --check-prefix=SINCOS-GNU

; Combine sin / cos into a single call unless they may write errno (as
; captured by readnone attrbiute, controlled by clang -fmath-errno
; setting).
; rdar://12856873

define float @test1(float %x) nounwind {
entry:
; SINCOS-LABEL: test1:
; SINCOS: bl ___sincosf_stret

; SINCOS-GNU-LABEL: test1:
; SINCOS-GNU: bl sincosf

; NOOPT-LABEL: test1:
; NOOPT: bl _sinf
; NOOPT: bl _cosf

  %call = tail call float @sinf(float %x) readnone
  %call1 = tail call float @cosf(float %x) readnone
  %add = fadd float %call, %call1
  ret float %add
}

define float @test1_errno(float %x) nounwind {
entry:
; SINCOS-LABEL: test1_errno:
; SINCOS: bl _sinf
; SINCOS: bl _cosf

; SINCOS-GNU-LABEL: test1_errno:
; SINCOS-GNU: bl sinf
; SINCOS-GNU: bl cosf

; NOOPT-LABEL: test1_errno:
; NOOPT: bl _sinf
; NOOPT: bl _cosf

  %call = tail call float @sinf(float %x)
  %call1 = tail call float @cosf(float %x)
  %add = fadd float %call, %call1
  ret float %add
}

define double @test2(double %x) nounwind {
entry:
; SINCOS-LABEL: test2:
; SINCOS: bl ___sincos_stret

; SINCOS-GNU-LABEL: test2:
; SINCOS-GNU: bl sincos

; NOOPT-LABEL: test2:
; NOOPT: bl _sin
; NOOPT: bl _cos

  %call = tail call double @sin(double %x) readnone
  %call1 = tail call double @cos(double %x) readnone
  %add = fadd double %call, %call1
  ret double %add
}

define double @test2_errno(double %x) nounwind {
entry:
; SINCOS-LABEL: test2_errno:
; SINCOS: bl _sin
; SINCOS: bl _cos

; SINCOS-GNU-LABEL: test2_errno:
; SINCOS-GNU: bl sin
; SINCOS-GNU: bl cos

; NOOPT-LABEL: test2_errno:
; NOOPT: bl _sin
; NOOPT: bl _cos

  %call = tail call double @sin(double %x)
  %call1 = tail call double @cos(double %x)
  %add = fadd double %call, %call1
  ret double %add
}

declare float  @sinf(float)
declare double @sin(double)
declare float @cosf(float)
declare double @cos(double)
