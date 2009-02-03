; RUN: llvm-as < %s | llc -march=x86-64 -mattr=+mmx | grep pxor
; RUN: llvm-as < %s | llc -march=x86-64 -mattr=+mmx | grep punpckldq

	%struct.vS1024 = type { [8 x <4 x i32>] }
	%struct.vS512 = type { [4 x <4 x i32>] }

declare <1 x i64> @llvm.x86.mmx.psrli.q(<1 x i64>, i32) nounwind readnone

define void @t() nounwind {
entry:
	br label %bb554

bb554:		; preds = %bb554, %entry
	%sum.0.reg2mem.0 = phi <1 x i64> [ %tmp562, %bb554 ], [ zeroinitializer, %entry ]		; <<1 x i64>> [#uses=1]
	%0 = load <1 x i64>* null, align 8		; <<1 x i64>> [#uses=2]
	%1 = bitcast <1 x i64> %0 to <2 x i32>		; <<2 x i32>> [#uses=1]
	%tmp555 = and <2 x i32> %1, < i32 -1, i32 0 >		; <<2 x i32>> [#uses=1]
	%2 = bitcast <2 x i32> %tmp555 to <1 x i64>		; <<1 x i64>> [#uses=1]
	%3 = call <1 x i64> @llvm.x86.mmx.psrli.q(<1 x i64> %0, i32 32) nounwind readnone		; <<1 x i64>> [#uses=1]
	%tmp558 = add <1 x i64> %sum.0.reg2mem.0, %2		; <<1 x i64>> [#uses=1]
	%4 = call <1 x i64> @llvm.x86.mmx.psrli.q(<1 x i64> %tmp558, i32 32) nounwind readnone		; <<1 x i64>> [#uses=1]
	%tmp562 = add <1 x i64> %4, %3		; <<1 x i64>> [#uses=1]
	br label %bb554
}
