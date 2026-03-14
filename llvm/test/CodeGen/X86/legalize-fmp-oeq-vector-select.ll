; RUN: llc -mtriple=x86_64-- -enable-legalize-types-checking < %s
; PR5092

define <4 x float> @bug(float %a) nounwind {
entry:
  %cmp = fcmp oeq float %a, 0.000000e+00          ; <i1> [#uses=1]
  %temp = select i1 %cmp, <4 x float> <float 1.000000e+00, float 0.000000e+00,
float 0.000000e+00, float 0.000000e+00>, <4 x float> zeroinitializer
  ret <4 x float> %temp
}

