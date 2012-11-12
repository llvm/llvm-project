; RUN: llc < %s -march=arm -mattr=+neon | FileCheck %s

; CHECK: t1
; CHECK: vldr d
; CHECK: vldr d
; CHECK: vadd.i16 d
; CHECK: vstr d
define void @t1(<2 x i32>* %r, <4 x i16>* %a, <4 x i16>* %b) nounwind {
entry:
	%0 = load <4 x i16>* %a, align 8		; <<4 x i16>> [#uses=1]
	%1 = load <4 x i16>* %b, align 8		; <<4 x i16>> [#uses=1]
	%2 = add <4 x i16> %0, %1		; <<4 x i16>> [#uses=1]
	%3 = bitcast <4 x i16> %2 to <2 x i32>		; <<2 x i32>> [#uses=1]
	store <2 x i32> %3, <2 x i32>* %r, align 8
	ret void
}

; CHECK: t2
; CHECK: vldr d
; CHECK: vldr d
; CHECK: vsub.i16 d
; CHECK: vmov r0, r1, d
define <2 x i32> @t2(<4 x i16>* %a, <4 x i16>* %b) nounwind readonly {
entry:
	%0 = load <4 x i16>* %a, align 8		; <<4 x i16>> [#uses=1]
	%1 = load <4 x i16>* %b, align 8		; <<4 x i16>> [#uses=1]
	%2 = sub <4 x i16> %0, %1		; <<4 x i16>> [#uses=1]
	%3 = bitcast <4 x i16> %2 to <2 x i32>		; <<2 x i32>> [#uses=1]
	ret <2 x i32> %3
}
