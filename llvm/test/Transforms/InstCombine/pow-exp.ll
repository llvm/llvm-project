; RUN: opt < %s -instcombine -S | FileCheck %s

define double @mypow(double %x, double %y) #0 {
entry:
  %call = call double @exp(double %x)
  %pow = call double @llvm.pow.f64(double %call, double %y)
  ret double %pow
}

; CHECK-LABEL: define double @mypow(
; CHECK:   %mul = fmul fast double %x, %y
; CHECK:   %exp = call double @exp(double %mul) #0
; CHECK:   ret double %exp
; CHECK: }

declare double @exp(double) #1
declare double @llvm.pow.f64(double, double)
attributes #0 = { "unsafe-fp-math"="true" }
attributes #1 = { "unsafe-fp-math"="true" }
