; RUN: opt < %s -instcombine -S | FileCheck %s

define double @log_pow(double %x, double %y) {
  %pow = call fast double @llvm.pow.f64(double %x, double %y)
  %call = call fast double @log(double %pow)
  ret double %call
}

; CHECK-LABEL: define double @log_pow(
; CHECK-NEXT:  %log = call fast double @log(double %x)
; CHECK-NEXT:  %mul = fmul fast double %log, %y
; CHECK-NEXT:  ret double %mul

define double @log_pow_not_fast(double %x, double %y) {
  %pow = call double @llvm.pow.f64(double %x, double %y)
  %call = call fast double @log(double %pow)
  ret double %call
}

; CHECK-LABEL: define double @log_pow_not_fast(
; CHECK-NEXT:  %pow = call double @llvm.pow.f64(double %x, double %y)
; CHECK-NEXT:  %call = call fast double @log(double %pow)
; CHECK-NEXT:  ret double %call

define double @function_pointer(double ()* %fptr, double %p1) {
  %call1 = call double %fptr()
  %pow = call double @log(double %call1)
  ret double %pow
}

; CHECK-LABEL: @function_pointer
; CHECK-NEXT:  %call1 = call double %fptr()
; CHECK-NEXT:  %pow = call double @log(double %call1)
; CHECK-NEXT:  ret double %pow

define double @log_exp2(double %x) {
  %call2 = call fast double @exp2(double %x)
  %call3 = call fast double @log(double %call2)
  ret double %call3
}

; CHECK-LABEL: @log_exp2
; CHECK-NEXT:  %call2 = call fast double @exp2(double %x)
; CHECK-NEXT:  %logmul = fmul fast double %x, 0x3FE62E42FEFA39EF
; CHECK-NEXT:  ret double %logmul

define double @log_exp2_not_fast(double %x) {
  %call2 = call double @exp2(double %x)
  %call3 = call fast double @log(double %call2)
  ret double %call3
}

; CHECK-LABEL: @log_exp2_not_fast
; CHECK-NEXT:  %call2 = call double @exp2(double %x)
; CHECK-NEXT:  %call3 = call fast double @log(double %call2)
; CHECK-NEXT:  ret double %call3

declare double @log(double) #0
declare double @exp2(double)
declare double @llvm.pow.f64(double, double)

attributes #0 = { nounwind readnone }
