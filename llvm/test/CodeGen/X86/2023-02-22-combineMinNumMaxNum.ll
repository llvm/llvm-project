; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mcpu=skylake

; Checking for a DAGCombine ICE.

define float @test_combinemaxnum(float %sub) #0 {
L.entry:
  %maxnum1 = call float @llvm.maxnum.f32(float 0.000000e+00, float 0.000000e+00)
  br label %L.LB21_850

L.LB21_850:
  %neg1 = fneg fast float %maxnum1
  %neg2 = fneg fast float %sub
  %mask = fcmp fast ule float %maxnum1, %neg2
  %maxnum2 = select i1 %mask, float %neg1, float %sub
  ret float %maxnum2
}

declare float @llvm.maxnum.f32(float, float) 

attributes #0 = { "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" }
