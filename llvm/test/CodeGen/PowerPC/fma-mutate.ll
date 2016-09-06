; Test several VSX FMA mutation opportunities.  The first one isn't a
; reasonable transformation because the killed product register is the
; same as the FMA target register.  The second one is legal.  The third
; one doesn't fit the feeding-copy pattern.

; RUN: llc -verify-machineinstrs < %s -mtriple=powerpc64-unknown-linux-gnu -mcpu=pwr7 -enable-unsafe-fp-math -mattr=+vsx -disable-ppc-vsx-fma-mutation=false | FileCheck %s
target datalayout = "E-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-f128:128:128-v128:128:128-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

declare double @llvm.sqrt.f64(double)

define double @foo3(double %a) nounwind {
  %r = call double @llvm.sqrt.f64(double %a)
  ret double %r

; CHECK: @foo3
; CHECK: xsnmsubadp [[REG:[0-9]+]], {{[0-9]+}}, [[REG]]
; CHECK: xsmaddmdp
; CHECK: xsmaddadp
}

