; RUN: llc -mtriple=arm-eabi -float-abi=soft -mattr=+neon %s -o - | FileCheck %s

; CHECK: t1
; CHECK: vldr d
; CHECK: vldr d
; CHECK: vadd.i16 d
; CHECK: vstr d
define void @t1(ptr %r, ptr %a, ptr %b) nounwind {
entry:
	%0 = load <4 x i16>, ptr %a, align 8		; <<4 x i16>> [#uses=1]
	%1 = load <4 x i16>, ptr %b, align 8		; <<4 x i16>> [#uses=1]
	%2 = add <4 x i16> %0, %1		; <<4 x i16>> [#uses=1]
	%3 = bitcast <4 x i16> %2 to <2 x i32>		; <<2 x i32>> [#uses=1]
	store <2 x i32> %3, ptr %r, align 8
	ret void
}

; CHECK: t2
; CHECK: vldr d
; CHECK: vldr d
; CHECK: vsub.i16 d
; CHECK: vmov r0, r1, d
define <2 x i32> @t2(ptr %a, ptr %b) nounwind readonly {
entry:
	%0 = load <4 x i16>, ptr %a, align 8		; <<4 x i16>> [#uses=1]
	%1 = load <4 x i16>, ptr %b, align 8		; <<4 x i16>> [#uses=1]
	%2 = sub <4 x i16> %0, %1		; <<4 x i16>> [#uses=1]
	%3 = bitcast <4 x i16> %2 to <2 x i32>		; <<2 x i32>> [#uses=1]
	ret <2 x i32> %3
}
