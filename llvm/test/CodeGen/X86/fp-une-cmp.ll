; RUN: llc < %s -march=x86 -mattr=sse4.1 | FileCheck %s
; <rdar://problem/7859988>

; Make sure we don't generate more jumps than we need to. We used to generate
; something like this:
;
;       jne  LBB0_1
;       jnp  LBB0_2
;   LBB0_1:
;       jmp  LBB0_3
;   LBB0_2:
;       addsd ...
;   LBB0_3:
;
; Now we generate this:
;
;       jne  LBB0_2
;       jp   LBB0_2
;       addsd ...
;   LBB0_2:

; CHECK:       func
; CHECK:       jne [[LABEL:.*]]
; CHECK-NEXT:  jp  [[LABEL]]
; CHECK-NOT:   jmp

define float @func(float %x, float %y) nounwind readnone optsize ssp {
entry:
  %0 = fpext float %x to double
  %1 = fpext float %y to double
  %2 = fmul double %0, %1
  %3 = fcmp une double %2, 0.000000e+00
  br i1 %3, label %bb2, label %bb1

bb1:
  %4 = fadd double %2, -1.000000e+00
  br label %bb2

bb2:
  %.0.in = phi double [ %4, %bb1 ], [ %2, %entry ]
  %.0 = fptrunc double %.0.in to float
  ret float %.0
}
