; RUN: llc < %s -mtriple=x86_64-apple-darwin -mcpu=corei7-avx -mattr=+avx | FileCheck %s

; CHECK: vmovsldup
define <8 x float> @movdupA(<8 x float> %src) nounwind uwtable readnone ssp {
entry:
  %shuffle.i = shufflevector <8 x float> %src, <8 x float> undef, <8 x i32> <i32 0, i32 0, i32 2, i32 2, i32 4, i32 4, i32 6, i32 6>
  ret <8 x float> %shuffle.i
}

; CHECK: vmovshdup
define <8 x float> @movdupB(<8 x float> %src) nounwind uwtable readnone ssp {
entry:
  %shuffle.i = shufflevector <8 x float> %src, <8 x float> undef, <8 x i32> <i32 1, i32 1, i32 3, i32 3, i32 5, i32 5, i32 7, i32 7>
  ret <8 x float> %shuffle.i
}

; CHECK: vmovsldup
define <4 x i64> @movdupC(<4 x i64> %src) nounwind uwtable readnone ssp {
entry:
  %0 = bitcast <4 x i64> %src to <8 x float>
  %shuffle.i = shufflevector <8 x float> %0, <8 x float> undef, <8 x i32> <i32 0, i32 0, i32 2, i32 2, i32 4, i32 4, i32 6, i32 6>
  %1 = bitcast <8 x float> %shuffle.i to <4 x i64>
  ret <4 x i64> %1
}

; CHECK: vmovshdup
define <4 x i64> @movdupD(<4 x i64> %src) nounwind uwtable readnone ssp {
entry:
  %0 = bitcast <4 x i64> %src to <8 x float>
  %shuffle.i = shufflevector <8 x float> %0, <8 x float> undef, <8 x i32> <i32 1, i32 1, i32 3, i32 3, i32 5, i32 5, i32 7, i32 7>
  %1 = bitcast <8 x float> %shuffle.i to <4 x i64>
  ret <4 x i64> %1
}

