; REQUIRES: asserts
; RUN: llc < %s -mtriple=i386-linux-gnu -mcpu=penryn -mattr=+sse2 -stats 2>&1 | \
; RUN:     grep "asm-printer" | grep 33

target datalayout = "e-p:32:32"
define void @foo(ptr %mc, ptr %bp, ptr %ms, ptr %xmb, ptr %mpp, ptr %tpmm, ptr %ip, ptr %tpim, ptr %dpp, ptr %tpdm, ptr %bpi, i32 %M) nounwind {
entry:
	%tmp9 = icmp slt i32 %M, 5		; <i1> [#uses=1]
	br i1 %tmp9, label %return, label %cond_true

cond_true:		; preds = %cond_true, %entry
	%indvar = phi i32 [ 0, %entry ], [ %indvar.next, %cond_true ]		; <i32> [#uses=2]
	%tmp. = shl i32 %indvar, 2		; <i32> [#uses=1]
	%tmp.10 = add nsw i32 %tmp., 1		; <i32> [#uses=2]
	%tmp31 = add nsw i32 %tmp.10, -1		; <i32> [#uses=4]
	%tmp32 = getelementptr i32, ptr %mpp, i32 %tmp31		; <ptr> [#uses=1]
	%tmp = load <16 x i8>, ptr %tmp32, align 1
	%tmp42 = getelementptr i32, ptr %tpmm, i32 %tmp31		; <ptr> [#uses=1]
	%tmp46 = load <4 x i32>, ptr %tmp42		; <<4 x i32>> [#uses=1]
	%tmp54 = bitcast <16 x i8> %tmp to <4 x i32>		; <<4 x i32>> [#uses=1]
	%tmp55 = add <4 x i32> %tmp54, %tmp46		; <<4 x i32>> [#uses=2]
	%tmp55.upgrd.2 = bitcast <4 x i32> %tmp55 to <2 x i64>		; <<2 x i64>> [#uses=1]
	%tmp62 = getelementptr i32, ptr %ip, i32 %tmp31		; <ptr> [#uses=1]
	%tmp66 = load <16 x i8>, ptr %tmp62, align 1
	%tmp73 = getelementptr i32, ptr %tpim, i32 %tmp31		; <ptr> [#uses=1]
	%tmp77 = load <4 x i32>, ptr %tmp73		; <<4 x i32>> [#uses=1]
	%tmp87 = bitcast <16 x i8> %tmp66 to <4 x i32>		; <<4 x i32>> [#uses=1]
	%tmp88 = add <4 x i32> %tmp87, %tmp77		; <<4 x i32>> [#uses=2]
	%tmp88.upgrd.4 = bitcast <4 x i32> %tmp88 to <2 x i64>		; <<2 x i64>> [#uses=1]
	%tmp99 = tail call <4 x i32> @llvm.x86.sse2.psra.d( <4 x i32> %tmp88, <4 x i32> %tmp55 )		; <<4 x i32>> [#uses=1]
	%tmp99.upgrd.5 = bitcast <4 x i32> %tmp99 to <2 x i64>		; <<2 x i64>> [#uses=2]
	%tmp110 = xor <2 x i64> %tmp99.upgrd.5, < i64 -1, i64 -1 >		; <<2 x i64>> [#uses=1]
	%tmp111 = and <2 x i64> %tmp110, %tmp55.upgrd.2		; <<2 x i64>> [#uses=1]
	%tmp121 = and <2 x i64> %tmp99.upgrd.5, %tmp88.upgrd.4		; <<2 x i64>> [#uses=1]
	%tmp131 = or <2 x i64> %tmp121, %tmp111		; <<2 x i64>> [#uses=1]
	%tmp137 = getelementptr i32, ptr %mc, i32 %tmp.10		; <ptr> [#uses=1]
	store <2 x i64> %tmp131, ptr %tmp137
	%tmp147 = add nsw i32 %tmp.10, 8		; <i32> [#uses=1]
	%tmp.upgrd.8 = icmp ne i32 %tmp147, %M		; <i1> [#uses=1]
	%indvar.next = add i32 %indvar, 1		; <i32> [#uses=1]
	br i1 %tmp.upgrd.8, label %cond_true, label %return

return:		; preds = %cond_true, %entry
	ret void
}

declare <4 x i32> @llvm.x86.sse2.psra.d(<4 x i32>, <4 x i32>)
