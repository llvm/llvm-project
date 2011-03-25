; RUN: llc < %s -mtriple=x86_64-linux | FileCheck %s
; CHECK-NOT:     mov
; CHECK:     paddw
; CHECK-NOT:     mov
; CHECK:     paddw
; CHECK-NOT:     paddw
; CHECK-NOT:     mov

; The 2-addr pass should ensure that identical code is produced for these functions
; no extra copy should be generated.

define <2 x i64> @test1(<2 x i64> %x, <2 x i64> %y) nounwind  {
entry:
	%tmp6 = bitcast <2 x i64> %y to <8 x i16>		; <<8 x i16>> [#uses=1]
	%tmp8 = bitcast <2 x i64> %x to <8 x i16>		; <<8 x i16>> [#uses=1]
	%tmp9 = add <8 x i16> %tmp8, %tmp6		; <<8 x i16>> [#uses=1]
	%tmp10 = bitcast <8 x i16> %tmp9 to <2 x i64>		; <<2 x i64>> [#uses=1]
	ret <2 x i64> %tmp10
}

define <2 x i64> @test2(<2 x i64> %x, <2 x i64> %y) nounwind  {
entry:
	%tmp6 = bitcast <2 x i64> %x to <8 x i16>		; <<8 x i16>> [#uses=1]
	%tmp8 = bitcast <2 x i64> %y to <8 x i16>		; <<8 x i16>> [#uses=1]
	%tmp9 = add <8 x i16> %tmp8, %tmp6		; <<8 x i16>> [#uses=1]
	%tmp10 = bitcast <8 x i16> %tmp9 to <2 x i64>		; <<2 x i64>> [#uses=1]
	ret <2 x i64> %tmp10
}


; The coalescer should commute the add to avoid a copy.
define <4 x float> @test3(<4 x float> %V) {
entry:
        %tmp8 = shufflevector <4 x float> %V, <4 x float> undef,
                                        <4 x i32> < i32 3, i32 2, i32 1, i32 0 >
        %add = fadd <4 x float> %tmp8, %V
        ret <4 x float> %add
}

