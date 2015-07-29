; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mcpu=x86-64 | FileCheck %s

; Anything more than one division using a single divisor operand
; should be converted into a reciprocal and multiplication.

define float @div1_arcp(float %x, float %y, float %z) #0 {
; CHECK-LABEL: div1_arcp:
; CHECK:       # BB#0:
; CHECK-NEXT:    divss %xmm1, %xmm0
; CHECK-NEXT:    retq
  %div1 = fdiv arcp float %x, %y
  ret float %div1
}

define float @div2_arcp(float %x, float %y, float %z) #0 {
; CHECK-LABEL: div2_arcp:
; CHECK:       # BB#0:
; CHECK-NEXT:    movss {{.*#+}} xmm3 = mem[0],zero,zero,zero
; CHECK-NEXT:    divss %xmm2, %xmm3
; CHECK-NEXT:    mulss %xmm1, %xmm0
; CHECK-NEXT:    mulss %xmm3, %xmm0
; CHECK-NEXT:    mulss %xmm3, %xmm0
; CHECK-NEXT:    retq
  %div1 = fdiv arcp float %x, %z
  %mul = fmul arcp float %div1, %y
  %div2 = fdiv arcp float %mul, %z
  ret float %div2
}

; If the reciprocal is already calculated, we should not
; generate an extra multiplication by 1.0. 

define double @div3_arcp(double %x, double %y, double %z) #0 {
; CHECK-LABEL: div3_arcp:
; CHECK:       # BB#0:
; CHECK-NEXT:    movsd{{.*#+}} xmm2 = mem[0],zero
; CHECK-NEXT:    divsd %xmm1, %xmm2
; CHECK-NEXT:    mulsd %xmm2, %xmm0
; CHECK-NEXT:    addsd %xmm2, %xmm0
; CHECK-NEXT:    retq
  %div1 = fdiv fast double 1.0, %y
  %div2 = fdiv fast double %x, %y
  %ret = fadd fast double %div2, %div1
  ret double %ret
}

define void @PR24141() #0 {
; CHECK-LABEL: PR24141:
; CHECK:	callq
; CHECK-NEXT:	divsd
; CHECK-NEXT:	jmp
entry:
  br label %while.body

while.body:
  %x.0 = phi double [ undef, %entry ], [ %div, %while.body ]
  %call = call { double, double } @g(double %x.0)
  %xv0 = extractvalue { double, double } %call, 0
  %xv1 = extractvalue { double, double } %call, 1
  %div = fdiv double %xv0, %xv1
  br label %while.body
}

declare { double, double } @g(double)

; FIXME: If the backend understands 'arcp', then this attribute is unnecessary.
attributes #0 = { "unsafe-fp-math"="true" }
