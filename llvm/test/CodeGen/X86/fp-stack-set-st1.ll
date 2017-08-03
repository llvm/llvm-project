; RUN: llc < %s -march=x86 | grep fxch | count 2

define i32 @main() nounwind {
entry:
	%asmtmp = tail call { double, double } asm sideeffect "fmul\09%st(1),%st\0A\09fst\09%st(1)\0A\09frndint\0A\09fxch  %st(1)\0A\09fsub\09%st(1),%st\0A\09f2xm1\0A\09", "={st},={st(1)},0,1,~{dirflag},~{fpsr},~{flags}"(double 0x4030FEFBD582097D, double 4.620000e+01) nounwind		; <{ double, double }> [#uses=0]
	unreachable
}
