; RUN: not llc -mtriple=armv8-eabi -mattr=+neon %s -o /dev/null 2>&1 | FileCheck %s

; CHECK: inline assembly requires more registers than available
define <4 x float> @t-constraint-float-vectors-too-few-regs(<4 x float> %a, <4 x float> %b) {
entry:
	%0 = tail call { <4 x float>, <4 x float>, <4 x float>, <4 x float>, <4 x float>, <4 x float>, <4 x float>, <4 x float>, <4 x float>, <4 x float> } asm "vadd.F32 $0, $9, $10\0A\09vadd.F32 $1, $9, $10\0A\09vadd.F32 $2, $9, $10\0A\09vadd.F32 $3, $9, $10\0A\09vadd.F32 $4, $9, $10\0A\09vadd.F32 $5, $9, $10\0A\09vadd.F32 $6, $9, $10\0A\09vadd.F32 $7, $9, $10\0A\09vadd.F32 $8, $9, $10", "=t,=t,=t,=t,=t,=t,=t,=t,=t,=t,t,t"(<4 x float> %a, <4 x float> %b)
	%asmresult = extractvalue { <4 x float>, <4 x float>, <4 x float>, <4 x float>, <4 x float>, <4 x float>, <4 x float>, <4 x float>, <4 x float>, <4 x float> } %0, 0
	ret <4 x float> %asmresult
}
