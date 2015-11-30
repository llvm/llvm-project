; RUN: opt < %s -instcombine -S | FileCheck %s

define double @mylog(double %x, double %y) #0 {
entry:
  %pow = call double @llvm.pow.f64(double %x, double %y)
  %call = call double @log(double %pow) #0
  ret double %call
}

; CHECK-LABEL: define double @mylog(
; CHECK:   %pow = call double @llvm.pow.f64(double %x, double %y)
; CHECK:   %call = call double @log(double %pow)
; CHECK:   ret double %call
; CHECK: }

declare double @log(double) #0
declare double @llvm.pow.f64(double, double)
