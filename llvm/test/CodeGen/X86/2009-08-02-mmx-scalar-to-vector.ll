; RUN: llc < %s -march=x86-64
; PR4669
declare x86_mmx @llvm.x86.mmx.pslli.q(x86_mmx, i32)

define <1 x i64> @test(i64 %t) {
entry:
	%t1 = insertelement <1 x i64> undef, i64 %t, i32 0
        %t0 = bitcast <1 x i64> %t1 to x86_mmx
	%t2 = tail call x86_mmx @llvm.x86.mmx.pslli.q(x86_mmx %t0, i32 48)
        %t3 = bitcast x86_mmx %t2 to <1 x i64>
	ret <1 x i64> %t3
}
