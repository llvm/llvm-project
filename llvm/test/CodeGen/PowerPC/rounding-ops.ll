; RUN: llc -verify-machineinstrs -mtriple=powerpc64-unknown-linux-gnu -mcpu=pwr7 -mattr=-vsx < %s | FileCheck %s
; RUN: llc -verify-machineinstrs -mtriple=powerpc64-unknown-linux-gnu -mcpu=pwr7 -mattr=+vsx < %s | FileCheck -check-prefix=CHECK-VSX %s
target datalayout = "E-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-f128:128:128-v128:128:128-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

define float @test1(float %x) nounwind  {
  %call = tail call float @floorf(float %x) nounwind readnone
  ret float %call

; CHECK-LABEL: test1:
; CHECK: frim 1, 1
; CHECK-VSX-LABEL: test1:
; CHECK-VSX: xsrdpim 1, 1
}

declare float @floorf(float) nounwind readnone

define double @test2(double %x) nounwind  {
  %call = tail call double @floor(double %x) nounwind readnone
  ret double %call

; CHECK-LABEL: test2:
; CHECK: frim 1, 1
; CHECK-VSX-LABEL: test2:
; CHECK-VSX: xsrdpim 1, 1
}

declare double @floor(double) nounwind readnone

define float @test3(float %x) nounwind  {
  %call = tail call float @roundf(float %x) nounwind readnone
  ret float %call

; CHECK-LABEL: test3:
; CHECK: frin 1, 1
; CHECK-VSX-LABEL: test3:
; CHECK-VSX: xsrdpi 1, 1
}

declare float @roundf(float) nounwind readnone

define double @test4(double %x) nounwind  {
  %call = tail call double @round(double %x) nounwind readnone
  ret double %call

; CHECK-LABEL: test4:
; CHECK: frin 1, 1
; CHECK-VSX-LABEL: test4:
; CHECK-VSX: xsrdpi 1, 1
}

declare double @round(double) nounwind readnone

define float @test5(float %x) nounwind  {
  %call = tail call float @ceilf(float %x) nounwind readnone
  ret float %call

; CHECK-LABEL: test5:
; CHECK: frip 1, 1
; CHECK-VSX-LABEL: test5:
; CHECK-VSX: xsrdpip 1, 1
}

declare float @ceilf(float) nounwind readnone

define double @test6(double %x) nounwind  {
  %call = tail call double @ceil(double %x) nounwind readnone
  ret double %call

; CHECK-LABEL: test6:
; CHECK: frip 1, 1
; CHECK-VSX-LABEL: test6:
; CHECK-VSX: xsrdpip 1, 1
}

declare double @ceil(double) nounwind readnone

define float @test9(float %x) nounwind  {
  %call = tail call float @truncf(float %x) nounwind readnone
  ret float %call

; CHECK-LABEL: test9:
; CHECK: friz 1, 1
; CHECK-VSX-LABEL: test9:
; CHECK-VSX: xsrdpiz 1, 1
}

declare float @truncf(float) nounwind readnone

define double @test10(double %x) nounwind  {
  %call = tail call double @trunc(double %x) nounwind readnone
  ret double %call

; CHECK-LABEL: test10:
; CHECK: friz 1, 1
; CHECK-VSX-LABEL: test10:
; CHECK-VSX: xsrdpiz 1, 1
}

declare double @trunc(double) nounwind readnone

