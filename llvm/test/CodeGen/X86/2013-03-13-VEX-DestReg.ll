; RUN: llc < %s -mtriple=x86_64-apple-macosx10.8.0 -mcpu=core-avx-i -show-mc-encoding

; ModuleID = 'bugpoint-reduced-simplified.bc'
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.8.0"

@b = external global [8 x float], align 32
@e = external global [8 x float], align 16

define void @main() #0 {
entry:
  %0 = load <8 x float>, <8 x float>* bitcast ([8 x float]* @b to <8 x float>*), align 32
  %bitcast.i = extractelement <8 x float> %0, i32 0
  %vecinit.i.i = insertelement <4 x float> undef, float %bitcast.i, i32 0
  %vecinit2.i.i = insertelement <4 x float> %vecinit.i.i, float 0.000000e+00, i32 1
  %vecinit3.i.i = insertelement <4 x float> %vecinit2.i.i, float 0.000000e+00, i32 2
  %vecinit4.i.i = insertelement <4 x float> %vecinit3.i.i, float 0.000000e+00, i32 3
  %1 = tail call <4 x float> @llvm.x86.sse.rcp.ss(<4 x float> %vecinit4.i.i) #2
  %vecext.i.i = extractelement <4 x float> %1, i32 0
  store float %vecext.i.i, float* getelementptr inbounds ([8 x float], [8 x float]* @e, i64 0, i64 0), align 16
  unreachable
}

declare <4 x float> @llvm.x86.sse.rcp.ss(<4 x float>) #1

attributes #0 = { nounwind ssp uwtable "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }
attributes #2 = { nounwind }
