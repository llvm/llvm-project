; RUN: llc %s -march=x86 -mcpu=yonah -o - | grep mulsd | count 6
; Ideally this would compile to 5 multiplies.

define double @_Z3f10d(double %a) nounwind readonly ssp noredzone {
entry:
  %0 = tail call double @llvm.powi.f64(double %a, i32 15) nounwind ; <double> [#uses=1]
  ret double %0
}

declare double @llvm.powi.f64(double, i32) nounwind readonly

