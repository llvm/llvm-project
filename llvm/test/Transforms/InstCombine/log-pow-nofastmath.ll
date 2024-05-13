; RUN: opt < %s -passes=instcombine -S | FileCheck %s

define double @mylog(double %x, double %y) {
entry:
  %pow = call double @llvm.pow.f64(double %x, double %y)
  %call = call double @log(double %pow)
  ret double %call
}

; CHECK-LABEL: define double @mylog(
; CHECK:   %pow = call double @llvm.pow.f64(double %x, double %y)
; CHECK:   %call = call double @log(double %pow)
; CHECK:   ret double %call
; CHECK: }

define double @test3(double %x) {
  %call2 = call double @exp2(double %x)
  %call3 = call double @log(double %call2)
  ret double %call3
}

; CHECK-LABEL: @test3
; CHECK:   %call2 = call double @exp2(double %x)
; CHECK:   %call3 = call double @log(double %call2)
; CHECK:   ret double %call3
; CHECK: }

declare double @log(double)
declare double @exp2(double)
declare double @llvm.pow.f64(double, double)
