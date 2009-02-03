; All of these ands and shifts should be folded into rlwimi's
; RUN: llvm-as < %s | llc -march=ppc32 -o %t -f
; RUN: grep rlwimi %t | count 3
; RUN: grep srwi   %t | count 1
; RUN: not grep slwi %t

define i16 @test1(i32 %srcA, i32 %srcB, i32 %alpha) {
entry:
	%tmp.1 = shl i32 %srcA, 15		; <i32> [#uses=1]
	%tmp.4 = and i32 %tmp.1, 32505856		; <i32> [#uses=1]
	%tmp.6 = and i32 %srcA, 31775		; <i32> [#uses=1]
	%tmp.7 = or i32 %tmp.4, %tmp.6		; <i32> [#uses=1]
	%tmp.9 = shl i32 %srcB, 15		; <i32> [#uses=1]
	%tmp.12 = and i32 %tmp.9, 32505856		; <i32> [#uses=1]
	%tmp.14 = and i32 %srcB, 31775		; <i32> [#uses=1]
	%tmp.15 = or i32 %tmp.12, %tmp.14		; <i32> [#uses=1]
	%tmp.18 = mul i32 %tmp.7, %alpha		; <i32> [#uses=1]
	%tmp.20 = sub i32 32, %alpha		; <i32> [#uses=1]
	%tmp.22 = mul i32 %tmp.15, %tmp.20		; <i32> [#uses=1]
	%tmp.23 = add i32 %tmp.22, %tmp.18		; <i32> [#uses=2]
	%tmp.27 = lshr i32 %tmp.23, 5		; <i32> [#uses=1]
	%tmp.28 = trunc i32 %tmp.27 to i16		; <i16> [#uses=1]
	%tmp.29 = and i16 %tmp.28, 31775		; <i16> [#uses=1]
	%tmp.33 = lshr i32 %tmp.23, 20		; <i32> [#uses=1]
	%tmp.34 = trunc i32 %tmp.33 to i16		; <i16> [#uses=1]
	%tmp.35 = and i16 %tmp.34, 992		; <i16> [#uses=1]
	%tmp.36 = or i16 %tmp.29, %tmp.35		; <i16> [#uses=1]
	ret i16 %tmp.36
}
