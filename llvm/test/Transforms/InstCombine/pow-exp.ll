; RUN: opt < %s -instcombine -S | FileCheck %s

define double @pow_exp(double %x, double %y) {
  %call = call fast double @exp(double %x) nounwind readnone
  %pow = call fast double @llvm.pow.f64(double %call, double %y)
  ret double %pow
}

; CHECK-LABEL: define double @pow_exp(
; CHECK-NEXT:  %mul = fmul fast double %x, %y
; CHECK-NEXT:  %exp = call fast double @exp(double %mul)
; CHECK-NEXT:  ret double %exp

define double @pow_exp2(double %x, double %y) {
  %call = call fast double @exp2(double %x) nounwind readnone
  %pow = call fast double @llvm.pow.f64(double %call, double %y)
  ret double %pow
}

; CHECK-LABEL: define double @pow_exp2(
; CHECK-NEXT:  %mul = fmul fast double %x, %y
; CHECK-NEXT:  %exp2 = call fast double @exp2(double %mul)
; CHECK-NEXT:  ret double %exp2

define double @pow_exp_not_fast(double %x, double %y) {
  %call = call double @exp(double %x)
  %pow = call fast double @llvm.pow.f64(double %call, double %y)
  ret double %pow
}

; CHECK-LABEL: define double @pow_exp_not_fast(
; CHECK-NEXT:  %call = call double @exp(double %x)
; CHECK-NEXT:  %pow = call fast double @llvm.pow.f64(double %call, double %y)
; CHECK-NEXT:  ret double %pow

define double @function_pointer(double ()* %fptr, double %p1) {
  %call1 = call fast double %fptr()
  %pow = call fast double @llvm.pow.f64(double %call1, double %p1)
  ret double %pow
}

; CHECK-LABEL: @function_pointer
; CHECK-NEXT:  %call1 = call fast double %fptr()
; CHECK-NEXT:  %pow = call fast double @llvm.pow.f64(double %call1, double %p1)

declare double @exp(double)
declare double @exp2(double)
declare double @llvm.pow.f64(double, double)

