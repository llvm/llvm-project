; RUN: llc -mtriple=arm-eabi -mattr=+neon %s -o /dev/null
; rdar://7770501 : Don't crash on SELECT and SELECT_CC with NEON vector values.

define void @vDSP_FFT16_copv(ptr nocapture %O, ptr nocapture %I, i32 %Direction) nounwind {
entry:
  %.22 = select i1 undef, <4 x float> undef, <4 x float> zeroinitializer ; <<4 x float>> [#uses=1]
  %0 = fadd <4 x float> undef, %.22               ; <<4 x float>> [#uses=1]
  %1 = fsub <4 x float> %0, undef                 ; <<4 x float>> [#uses=1]
  %2 = shufflevector <4 x float> %1, <4 x float> undef, <4 x i32> <i32 2, i32 6, i32 3, i32 7> ; <<4 x float>> [#uses=1]
  %3 = shufflevector <4 x float> undef, <4 x float> %2, <4 x i32> <i32 2, i32 3, i32 6, i32 7> ; <<4 x float>> [#uses=1]
  %4 = fmul <4 x float> %3, <float 0.000000e+00, float 0x3FED906BC0000000, float 0x3FE6A09E60000000, float 0xBFD87DE2A0000000> ; <<4 x float>> [#uses=1]
  %5 = fadd <4 x float> undef, %4                 ; <<4 x float>> [#uses=1]
  %6 = fadd <4 x float> undef, %5                 ; <<4 x float>> [#uses=1]
  %7 = fadd <4 x float> undef, %6                 ; <<4 x float>> [#uses=1]
  br i1 undef, label %bb4, label %bb3

bb3:                                              ; preds = %entry
  %8 = shufflevector <4 x float> undef, <4 x float> %7, <4 x i32> <i32 2, i32 6, i32 3, i32 7> ; <<4 x float>> [#uses=0]
  ret void

bb4:                                              ; preds = %entry
  ret void
}
