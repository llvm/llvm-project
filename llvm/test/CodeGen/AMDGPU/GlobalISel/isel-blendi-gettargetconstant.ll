; ModuleID = 'bugpoint-reduced-simplified.bc'
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-grtev4-linux-gnu"

define dso_local void @csrot_(float* %0) local_unnamed_addr #0 align 32 {
1:
  %2 = load float, float* %0, align 4
  %3 = fsub float -0.000000e+00, %2
  %4 = insertelement <2 x float> <float undef, float -0.000000e+00>, float %3, i32 0
  store <2 x float> %4, <2 x float>* undef, align 8
  ret void
}

attributes #0 = { "target-features"="+aes,+cx8,+fxsr,+mmx,+pclmul,+popcnt,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+ssse3,+x87" }
