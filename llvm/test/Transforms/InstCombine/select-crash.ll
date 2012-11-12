; RUN: opt < %s -instcombine -S | FileCheck %s
; Formerly crashed, PR8490.

define fastcc double @gimp_operation_color_balance_map(float %value, double %highlights) nounwind readnone inlinehint {
entry:
; CHECK: gimp_operation_color_balance_map
; CHECK: fsub double -0.000000
  %conv = fpext float %value to double
  %div = fdiv double %conv, 1.600000e+01
  %add = fadd double %div, 1.000000e+00
  %div1 = fdiv double 1.000000e+00, %add
  %sub = fsub double 1.075000e+00, %div1
  %sub24 = fsub double 1.000000e+00, %sub
  %add26 = fadd double %sub, 1.000000e+00
  %cmp86 = fcmp ogt double %highlights, 0.000000e+00
  %cond90 = select i1 %cmp86, double %sub24, double %add26
  %mul91 = fmul double %highlights, %cond90
  %add94 = fadd double undef, %mul91
  ret double %add94
}

; PR10180: same crash, but with vectors
define <4 x float> @foo(i1 %b, <4 x float> %x, <4 x float> %y, <4 x float> %z) {
; CHECK: @foo
; CHECK: fsub <4 x float>
; CHECK: select
; CHECK: fadd <4 x float>
  %a = fadd <4 x float> %x, %y
  %sub = fsub <4 x float> %x, %z
  %sel = select i1 %b, <4 x float> %a, <4 x float> %sub 
  ret <4 x float> %sel
}

; CHECK: @test3
define i32 @test3(i1 %bool, i32 %a) {
entry:
  %cond = or i1 %bool, true
  br i1 %cond, label %return, label %xpto

; technically reachable, but this malformed IR may appear as a result of constant propagation
xpto:
  %select = select i1 %bool, i32 %a, i32 %select
  %select2 = select i1 %bool, i32 %select2, i32 %a
  %sum = add i32 %select, %select2
  ret i32 %sum

return:
  ret i32 7
}
