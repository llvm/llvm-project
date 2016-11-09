; RUN: llc -mtriple=aarch64-none-linux-gnu -enable-unsafe-fp-math < %s
; There is no invocation to FileCheck as this
; caused a crash in "Post-RA pseudo instruction expansion"

define double @foo(float *%user, float %t17) {
  %t16 = load float, float* %user, align 8
  %conv = fpext float %t16 to double
  %cmp26 = fcmp fast oeq float %t17, 0.000000e+00
  %div = fdiv fast float %t16, %t17
  %div.op = fmul fast float %div, 1.000000e+02
  %t18 = fpext float %div.op to double
  %conv31 = select i1 %cmp26, double 0.000000e+00, double %t18
  ret double %conv31
}

