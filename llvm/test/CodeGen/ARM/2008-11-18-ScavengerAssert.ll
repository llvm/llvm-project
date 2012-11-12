; RUN: llc < %s -march=arm -mattr=+v6,+vfp2

define hidden i64 @__muldi3(i64 %u, i64 %v) nounwind {
entry:
	%0 = trunc i64 %u to i32		; <i32> [#uses=1]
	%asmtmp = tail call { i32, i32, i32, i32, i32 } asm "@ Inlined umul_ppmm\0A\09mov\09$2, $5, lsr #16\0A\09mov\09$0, $6, lsr #16\0A\09bic\09$3, $5, $2, lsl #16\0A\09bic\09$4, $6, $0, lsl #16\0A\09mul\09$1, $3, $4\0A\09mul\09$4, $2, $4\0A\09mul\09$3, $0, $3\0A\09mul\09$0, $2, $0\0A\09adds\09$3, $4, $3\0A\09addcs\09$0, $0, #65536\0A\09adds\09$1, $1, $3, lsl #16\0A\09adc\09$0, $0, $3, lsr #16", "=&r,=r,=&r,=&r,=r,r,r,~{cc}"(i32 %0, i32 0) nounwind		; <{ i32, i32, i32, i32, i32 }> [#uses=1]
	%asmresult1 = extractvalue { i32, i32, i32, i32, i32 } %asmtmp, 1		; <i32> [#uses=1]
	%asmresult116 = zext i32 %asmresult1 to i64		; <i64> [#uses=1]
	%asmresult116.ins = or i64 0, %asmresult116		; <i64> [#uses=1]
	%1 = lshr i64 %v, 32		; <i64> [#uses=1]
	%2 = mul i64 %1, %u		; <i64> [#uses=1]
	%3 = add i64 %2, 0		; <i64> [#uses=1]
	%4 = shl i64 %3, 32		; <i64> [#uses=1]
	%5 = add i64 %asmresult116.ins, %4		; <i64> [#uses=1]
	ret i64 %5
}
