; RUN: opt < %s -instcombine -S | FileCheck %s

define double @mylog(double %x, double %y) #0 {
entry:
  %pow = call double @llvm.pow.f64(double %x, double %y)
  %call = call double @log(double %pow) #0
  ret double %call
}

; CHECK-LABEL: define double @mylog(
; CHECK:   %log = call double @log(double %x) #0
; CHECK:   %mul = fmul fast double %log, %y
; CHECK:   ret double %mul
; CHECK: }

define double @test2(double ()* %fptr, double %p1) #0 {
  %call1 = call double %fptr()
  %pow = call double @log(double %call1)
  ret double %pow
}

; CHECK-LABEL: @test2
; CHECK: log

declare double @log(double) #0
declare double @llvm.pow.f64(double, double)

attributes #0 = { "unsafe-fp-math"="true" }
