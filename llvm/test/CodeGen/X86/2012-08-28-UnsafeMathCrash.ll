; RUN: llc < %s -enable-unsafe-fp-math
; <rdar://problem/12180135>
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128-n8:16:32-S128"
target triple = "i386-apple-macosx10.8.0"

define i32 @foo(float %mean) nounwind readnone ssp align 2 {
entry:
  %cmp = fcmp olt float %mean, -3.000000e+00
  %f.0 = select i1 %cmp, float -3.000000e+00, float %mean
  %cmp2 = fcmp ult float %f.0, 3.000000e+00
  %f.1 = select i1 %cmp2, float %f.0, float 0x4007EB8520000000
  %add = fadd float %f.1, 3.000000e+00
  %div = fdiv float %add, 2.343750e-02
  %0 = fpext float %div to double
  %conv = select i1 undef, double 2.550000e+02, double %0
  %add8 = fadd double %conv, 5.000000e-01
  %conv9 = fptosi double %add8 to i32
  %.conv9 = select i1 undef, i32 255, i32 %conv9
  ret i32 %.conv9
}
