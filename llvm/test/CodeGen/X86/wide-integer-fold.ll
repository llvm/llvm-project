; RUN: llc < %s -march=x86-64 | FileCheck %s
; CHECK:  movq  $-65535, %rax

; DAGCombiner should fold this to a simple constant.

define i64 @foo(i192 %a) nounwind {
  %t = or i192 %a, -22300404916163702203072254898040925442801665
  %s = and i192 %t, -22300404916163702203072254898040929737768960
  %u = lshr i192 %s, 128
  %v = trunc i192 %u to i64
  ret i64 %v
}
