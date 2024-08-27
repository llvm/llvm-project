; RUN: llc -mtriple=arm-eabi -mattr=+neon -float-abi=soft %s -o - | FileCheck %s

; CHECK: function1
; CHECK-NOT: vmov
define double @function1(double %a, double %b, double %c, double %d, double %e, double %f) nounwind noinline ssp {
entry:
  %call = tail call double @function2(double %f, double %e, double %d, double %c, double %b, double %a) nounwind
  ret double %call
}

declare double @function2(double, double, double, double, double, double)
