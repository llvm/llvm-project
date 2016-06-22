; RUN: llc < %s -mtriple=armv7-apple-ios6 -mcpu=cortex-a8 | FileCheck %s --check-prefix=NOOPT
; RUN: llc < %s -mtriple=armv7-apple-ios7 -mcpu=cortex-a8 | FileCheck %s --check-prefix=SINCOS
; RUN: llc < %s -mtriple=armv7-linux-gnu -mcpu=cortex-a8 | FileCheck %s --check-prefix=NOOPT-GNU
; RUN: llc < %s -mtriple=armv7-linux-gnueabi -mcpu=cortex-a8 \
; RUN:   --enable-unsafe-fp-math | FileCheck %s --check-prefix=SINCOS-GNU

; Combine sin / cos into a single call.
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

; NOOPT-GNU-LABEL: test1:
; NOOPT-GNU: bl sinf
; NOOPT-GNU: bl cosf

  %call = tail call float @sinf(float %x) nounwind readnone
  %call1 = tail call float @cosf(float %x) nounwind readnone
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

; NOOPT-GNU-LABEL: test2:
; NOOPT-GNU: bl sin
; NOOPT-GNU: bl cos
  %call = tail call double @sin(double %x) nounwind readnone
  %call1 = tail call double @cos(double %x) nounwind readnone
  %add = fadd double %call, %call1
  ret double %add
}

declare float  @sinf(float) readonly
declare double @sin(double) readonly
declare float @cosf(float) readonly
declare double @cos(double) readonly
