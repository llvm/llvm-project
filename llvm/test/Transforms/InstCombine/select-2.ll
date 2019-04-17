; RUN: opt < %s -instcombine -S | FileCheck %s

; CHECK: select
; CHECK: select

; Make sure instcombine don't fold select into operands. We don't want to emit
; select of two integers unless it's selecting 0 / 1.

define i32 @t1(i32 %c, i32 %x) nounwind {
       %t1 = icmp eq i32 %c, 0
       %t2 = lshr i32 %x, 18
       %t3 = select i1 %t1, i32 %t2, i32 %x
       ret i32 %t3
}

define i32 @t2(i32 %c, i32 %x) nounwind {
       %t1 = icmp eq i32 %c, 0
       %t2 = and i32 %x, 18
       %t3 = select i1 %t1, i32 %t2, i32 %x
       ret i32 %t3
}

define float @t3(float %x, float %y) nounwind {
  %t1 = fcmp ogt float %x, %y
  %t2 = select i1 %t1, float %x, float 1.0
  %t3 = fadd fast float %t2, 1.0
  ret float %t3
; CHECK-LABEL: @t3(
; CHECK: fadd fast
; CHECK: select
}
