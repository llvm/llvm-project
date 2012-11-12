; RUN: llc < %s -march=x86 -mcpu=penryn -mattr=+sse,-sse2 -mtriple=i386-apple-darwin | FileCheck %s -check-prefix=sse
; RUN: llc < %s -march=x86 -mcpu=penryn -mattr=+sse2 -mtriple=i386-apple-darwin | FileCheck %s -check-prefix=sse2

; sse:  t1:
; sse2: t1:
define <4 x float> @t1(<4 x float> %a, <4 x float> %b) nounwind  {
; sse: shufps
; sse2: pshufd
; sse2-NEXT: ret
        %tmp1 = shufflevector <4 x float> %b, <4 x float> undef, <4 x i32> zeroinitializer
        ret <4 x float> %tmp1
}

; sse:  t2:
; sse2: t2:
define <4 x float> @t2(<4 x float> %A, <4 x float> %B) nounwind {
; sse: shufps
; sse2: pshufd
; sse2-NEXT: ret
	%tmp = shufflevector <4 x float> %A, <4 x float> %B, <4 x i32> < i32 3, i32 3, i32 3, i32 3 >
	ret <4 x float> %tmp
}

; sse:  t3:
; sse2: t3:
define <4 x float> @t3(<4 x float> %A, <4 x float> %B) nounwind {
; sse: shufps
; sse2: pshufd
; sse2-NEXT: ret
	%tmp = shufflevector <4 x float> %A, <4 x float> %B, <4 x i32> < i32 4, i32 4, i32 4, i32 4 >
	ret <4 x float> %tmp
}

; sse:  t4:
; sse2: t4:
define <4 x float> @t4(<4 x float> %A, <4 x float> %B) nounwind {

; sse: shufps
; sse2: pshufd
; sse2-NEXT: ret
	%tmp = shufflevector <4 x float> %A, <4 x float> %B, <4 x i32> < i32 1, i32 3, i32 2, i32 0 >
	ret <4 x float> %tmp
}
