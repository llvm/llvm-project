; RUN: opt < %s -instcombine -S | FileCheck %s

define double @mylog(double %x, double %y) #0 {
entry:
  %pow = call double @llvm.pow.f64(double %x, double %y)
  %call = call double @log(double %pow) #0
  ret double %call
}

; CHECK-LABEL: define double @mylog(
; CHECK:   %log = call fast double @log(double %x) #0
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

define double @test3(double %x) #0 {
  %call2 = call double @exp2(double %x) #0
  %call3 = call double @log(double %call2) #0
  ret double %call3
}

; CHECK-LABEL: @test3
; CHECK:  %call2 = call double @exp2(double %x) #0
; CHECK:  %logmul = fmul fast double %x, 0x3FE62E42FEFA39EF
; CHECK:  ret double %logmul
; CHECK: }

declare double @log(double) #0
declare double @exp2(double) #0
declare double @llvm.pow.f64(double, double)

attributes #0 = { "unsafe-fp-math"="true" }
