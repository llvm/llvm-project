; RUN: opt < %s -reassociate -S | FileCheck %s
 
; CHECK-LABEL: func
; CHECK:       fmul fast double
; CHECK-NEXT:  fmul fast double
; CHECK-NEXT:  ret

define double @func(double %a, double %b) {
entry:
  %mul1 = fmul fast double %a, %a
  %mul2 = fmul fast double %b, %b
  %mul3 = fmul fast double %mul1, %mul2
  ret double %mul3
}
