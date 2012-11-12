; RUN: llc < %s -march=x86 -mcpu=penryn -mattr=sse41 | FileCheck %s

; ModuleID = 'vec_shuffle-27.bc'
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:32:32"
target triple = "i686-apple-cl.1.0"

define <8 x float> @my2filter4_1d(<4 x float> %a, <8 x float> %T0, <8 x float> %T1) nounwind readnone {
entry:
; CHECK: subps
; CHECK: mulps
; CHECK: addps
; CHECK: subps
; CHECK: mulps
; CHECK: addps
	%tmp7 = shufflevector <4 x float> %a, <4 x float> undef, <8 x i32> < i32 0, i32 1, i32 2, i32 3, i32 0, i32 1, i32 2, i32 3 >		; <<8 x float>> [#uses=1]
	%sub = fsub <8 x float> %T1, %T0		; <<8 x float>> [#uses=1]
	%mul = fmul <8 x float> %sub, %tmp7		; <<8 x float>> [#uses=1]
	%add = fadd <8 x float> %mul, %T0		; <<8 x float>> [#uses=1]
	ret <8 x float> %add
}

; Test case for r122206
define void @test2(<4 x i64>* %ap, <4 x i64>* %bp) nounwind {
entry:
; CHECK: movdqa
  %a = load <4 x i64> * %ap
  %b = load <4 x i64> * %bp
  %mulaa = mul <4 x i64> %a, %a
  %mulbb = mul <4 x i64> %b, %b
  %mulab = mul <4 x i64> %a, %b
  %vect1271 = shufflevector <4 x i64> %mulaa, <4 x i64> %mulbb, <4 x i32> <i32 0, i32 4, i32 undef, i32 undef>
  %vect1272 = shufflevector <4 x i64> %mulaa, <4 x i64> %mulbb, <4 x i32> <i32 1, i32 5, i32 undef, i32 undef>
  %vect1487 = shufflevector <4 x i64> %vect1271, <4 x i64> %mulab, <4 x i32> <i32 0, i32 1, i32 2, i32 4>
  %vect1488 = shufflevector <4 x i64> %vect1272, <4 x i64> %mulab, <4 x i32> <i32 0, i32 1, i32 2, i32 5>
  store <4 x i64> %vect1487, <4 x i64>* %ap
  store <4 x i64> %vect1488, <4 x i64>* %bp
  ret void;
}
