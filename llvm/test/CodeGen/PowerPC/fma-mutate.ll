; Test several VSX FMA mutation opportunities.  The first one isn't a
; reasonable transformation because the killed product register is the
; same as the FMA target register.  The second one is legal.  The third
; one doesn't fit the feeding-copy pattern.

; RUN: llc -verify-machineinstrs < %s -mtriple=powerpc64-unknown-linux-gnu -mcpu=pwr7 -mattr=+vsx -disable-ppc-vsx-fma-mutation=false | FileCheck %s
target datalayout = "E-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-f128:128:128-v128:128:128-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

declare double @llvm.sqrt.f64(double)

define double @foo3_fmf(double %a) nounwind {
; CHECK: @foo3_fmf
; CHECK-NOT: fmr
; CHECK: xsmaddmdp
; CHECK: xsmaddadp
  %r = call reassoc afn ninf double @llvm.sqrt.f64(double %a)
  ret double %r
}

define double @foo3_safe(double %a) nounwind {
; CHECK: @foo3_safe
; CHECK-NOT: fmr
; CHECK: xssqrtdp
  %r = call double @llvm.sqrt.f64(double %a)
  ret double %r
}

