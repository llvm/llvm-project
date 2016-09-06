; RUN: llc -verify-machineinstrs -mcpu=pwr7 -mtriple=powerpc64-unknown-linux-gnu -mattr=-vsx < %s | FileCheck %s
; RUN: llc -verify-machineinstrs -mcpu=pwr7 -mtriple=powerpc64-unknown-linux-gnu -mattr=+vsx < %s | FileCheck %s -check-prefix=CHECK-VSX
target datalayout = "E-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-f128:128:128-v128:128:128-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

define double @foo_d_ll(ppc_fp128 %a, ppc_fp128 %b) #0 {
entry:
  %call = tail call ppc_fp128 @copysignl(ppc_fp128 %a, ppc_fp128 %b) #0
  %conv = fptrunc ppc_fp128 %call to double
  ret double %conv

; CHECK-LABEL: @foo_d_ll
; CHECK: fcpsgn 1, 3, 1
; CHECK: blr
; CHECK-VSX-LABEL: @foo_d_ll
; CHECK-VSX: xscpsgndp 1, 3, 1
; CHECK-VSX: blr
}

declare ppc_fp128 @copysignl(ppc_fp128, ppc_fp128) #0

define double @foo_dl(double %a, ppc_fp128 %b) #0 {
entry:
  %conv = fptrunc ppc_fp128 %b to double
  %call = tail call double @copysign(double %a, double %conv) #0
  ret double %call

; CHECK-LABEL: @foo_dl
; CHECK: fcpsgn 1, 2, 1
; CHECK: blr
; CHECK-VSX-LABEL: @foo_dl
; CHECK-VSX: xscpsgndp 1, 2, 1
; CHECK-VSX: blr
}

declare double @copysign(double, double) #0

define ppc_fp128 @foo_ll(double %a, ppc_fp128 %b) #0 {
entry:
  %conv = fpext double %a to ppc_fp128
  %call = tail call ppc_fp128 @copysignl(ppc_fp128 %conv, ppc_fp128 %b) #0
  ret ppc_fp128 %call

; CHECK-LABEL: @foo_ll
; CHECK: bl copysignl
; CHECK: blr
; CHECK-VSX-LABEL: @foo_ll
; CHECK-VSX: bl copysignl
; CHECK-VSX: blr
}

define ppc_fp128 @foo_ld(double %a, double %b) #0 {
entry:
  %conv = fpext double %a to ppc_fp128
  %conv1 = fpext double %b to ppc_fp128
  %call = tail call ppc_fp128 @copysignl(ppc_fp128 %conv, ppc_fp128 %conv1) #0
  ret ppc_fp128 %call

; CHECK-LABEL: @foo_ld
; CHECK: bl copysignl
; CHECK: blr
; CHECK-VSX-LABEL: @foo_ld
; CHECK-VSX: bl copysignl
; CHECK-VSX: blr
}

define ppc_fp128 @foo_lf(double %a, float %b) #0 {
entry:
  %conv = fpext double %a to ppc_fp128
  %conv1 = fpext float %b to ppc_fp128
  %call = tail call ppc_fp128 @copysignl(ppc_fp128 %conv, ppc_fp128 %conv1) #0
  ret ppc_fp128 %call

; CHECK-LABEL: @foo_lf
; CHECK: bl copysignl
; CHECK: blr
; CHECK-VSX-LABEL: @foo_lf
; CHECK-VSX: bl copysignl
; CHECK-VSX: blr
}

attributes #0 = { nounwind readnone }

