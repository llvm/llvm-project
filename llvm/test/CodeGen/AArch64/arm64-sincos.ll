; RUN: llc < %s -mtriple=arm64-apple-ios7 | FileCheck %s --check-prefix CHECK-IOS
; RUN: llc < %s -mtriple=arm64-linux-gnu | FileCheck %s --check-prefix CHECK-LINUX

; Combine sin / cos into a single call unless they may write errno (as
; captured by readnone attrbiute, controlled by clang -fmath-errno
; setting).
; rdar://12856873

define float @test1(float %x) nounwind {
entry:
; CHECK-IOS-LABEL: test1:
; CHECK-IOS: bl ___sincosf_stret
; CHECK-IOS: fadd s0, s0, s1

; CHECK-LINUX-LABEL: test1:
; CHECK-LINUX: bl sincosf

  %call = tail call float @sinf(float %x) readnone
  %call1 = tail call float @cosf(float %x) readnone
  %add = fadd float %call, %call1
  ret float %add
}

define float @test1_errno(float %x) nounwind {
entry:
; CHECK-IOS-LABEL: test1_errno:
; CHECK-IOS: bl _sinf
; CHECK-IOS: bl _cosf

; CHECK-LINUX-LABEL: test1_errno:
; CHECK-LINUX: bl sinf
; CHECK-LINUX: bl cosf

  %call = tail call float @sinf(float %x)
  %call1 = tail call float @cosf(float %x)
  %add = fadd float %call, %call1
  ret float %add
}

define double @test2(double %x) nounwind {
entry:
; CHECK-IOS-LABEL: test2:
; CHECK-IOS: bl ___sincos_stret
; CHECK-IOS: fadd d0, d0, d1

; CHECK-LINUX-LABEL: test2:
; CHECK-LINUX: bl sincos

  %call = tail call double @sin(double %x) readnone
  %call1 = tail call double @cos(double %x) readnone
  %add = fadd double %call, %call1
  ret double %add
}

define double @test2_errno(double %x) nounwind {
entry:
; CHECK-IOS-LABEL: test2_errno:
; CHECK-IOS: bl _sin
; CHECK-IOS: bl _cos

; CHECK-LINUX-LABEL: test2_errno:
; CHECK-LINUX: bl sin
; CHECK-LINUX: bl cos

  %call = tail call double @sin(double %x)
  %call1 = tail call double @cos(double %x)
  %add = fadd double %call, %call1
  ret double %add
}

declare float  @sinf(float)
declare double @sin(double)
declare float @cosf(float)
declare double @cos(double)
