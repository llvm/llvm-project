; RUN: split-file %s %t
; RUN: not llvm-diff %t/a.ll %t/b.ll 2>&1 | FileCheck %s

; Regression test for https://github.com/llvm/llvm-project/issues/184133

; CHECK:      in function func:
; CHECK-NEXT:   in block %0 / %0:
; CHECK-NEXT:    >   %c = call double @h()
; CHECK-NEXT:    >   %b = call double @g(double %c)
; CHECK-NEXT:    >   ret double %b
; CHECK-NEXT:    <   %b = call double @g(double %a)
; CHECK-NEXT:    <   ret double %b
; CHECK-NOT: second time around

;--- a.ll
define double @func() {
  %a = call double @f()
  %b = call double @g(double %a)
  ret double %b
}
declare double @f()
declare double @g(double)

;--- b.ll
define double @func() {
  %a = call double @f()
  %c = call double @h()
  %b = call double @g(double %c)
  ret double %b
}
declare double @f()
declare double @h()
declare double @g(double)
