; RUN: llc -verify-machineinstrs -mtriple=powerpc64-unknown-linux-gnu -mcpu=pwr7 -mattr=-vsx < %s | FileCheck %s
; RUN: llc -verify-machineinstrs -mtriple=powerpc64-unknown-linux-gnu -mcpu=pwr7 -mattr=+vsx < %s | FileCheck -check-prefix=CHECK-VSX %s
target datalayout = "E-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-f128:128:128-v128:128:128-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

define double @foo_dd(double %a, double %b) #0 {
entry:
  %call = tail call double @copysign(double %a, double %b) #0
  ret double %call

; CHECK-LABEL: @foo_dd
; CHECK: fcpsgn 1, 2, 1
; CHECK: blr
; CHECK-VSX-LABEL: @foo_dd
; CHECK-VSX: xscpsgndp 1, 2, 1
; CHECK-VSX: blr
}

declare double @copysign(double, double) #0

define float @foo_ss(float %a, float %b) #0 {
entry:
  %call = tail call float @copysignf(float %a, float %b) #0
  ret float %call

; CHECK-LABEL: @foo_ss
; CHECK: fcpsgn 1, 2, 1
; CHECK: blr
; CHECK-VSX-LABEL: @foo_ss
; CHECK-VSX: fcpsgn 1, 2, 1
; CHECK-VSX: blr
}

declare float @copysignf(float, float) #0

define float @foo_sd(float %a, double %b) #0 {
entry:
  %conv = fptrunc double %b to float
  %call = tail call float @copysignf(float %a, float %conv) #0
  ret float %call

; CHECK-LABEL: @foo_sd
; CHECK: fcpsgn 1, 2, 1
; CHECK: blr
; CHECK-VSX-LABEL: @foo_sd
; CHECK-VSX: fcpsgn 1, 2, 1
; CHECK-VSX: blr
}

define double @foo_ds(double %a, float %b) #0 {
entry:
  %conv = fpext float %b to double
  %call = tail call double @copysign(double %a, double %conv) #0
  ret double %call

; CHECK-LABEL: @foo_ds
; CHECK: fcpsgn 1, 2, 1
; CHECK: blr
; CHECK-VSX-LABEL: @foo_ds
; CHECK-VSX: fcpsgn 1, 2, 1
; CHECK-VSX: blr
}

attributes #0 = { nounwind readnone }

