; RUN: llc < %s -march=x86 -mcpu=pentium-m  | FileCheck %s

define <4 x float> @t1(<4 x float> %a) nounwind  {
; CHECK: movlhps
  %tmp1 = shufflevector <4 x float> %a, <4 x float> undef, <4 x i32> < i32 0, i32 1, i32 0, i32 1 >       ; <<4 x float>> [#uses=1]
  ret <4 x float> %tmp1
}

define <4 x i32> @t2(<4 x i32>* %a) nounwind {
; CHECK: pshufd
; CHECK: ret
  %tmp1 = load <4 x i32>* %a
	%tmp2 = shufflevector <4 x i32> %tmp1, <4 x i32> undef, <4 x i32> < i32 0, i32 1, i32 0, i32 1 >		; <<4 x i32>> [#uses=1]
	ret <4 x i32> %tmp2
}
