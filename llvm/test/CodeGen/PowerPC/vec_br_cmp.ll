; RUN: llc -verify-machineinstrs < %s -mtriple=ppc32-- -mcpu=g5 -o %t
; RUN: grep vcmpeqfp. %t
; RUN: not grep mfcr %t

; A predicate compare used immediately by a branch should not generate an mfcr.

define void @test(ptr %A, ptr %B) {
	%tmp = load <4 x float>, ptr %A		; <<4 x float>> [#uses=1]
	%tmp3 = load <4 x float>, ptr %B		; <<4 x float>> [#uses=1]
	%tmp.upgrd.1 = tail call i32 @llvm.ppc.altivec.vcmpeqfp.p( i32 1, <4 x float> %tmp, <4 x float> %tmp3 )		; <i32> [#uses=1]
	%tmp.upgrd.2 = icmp eq i32 %tmp.upgrd.1, 0		; <i1> [#uses=1]
	br i1 %tmp.upgrd.2, label %cond_true, label %UnifiedReturnBlock

cond_true:		; preds = %0
	store <4 x float> zeroinitializer, ptr %B
	ret void

UnifiedReturnBlock:		; preds = %0
	ret void
}

declare i32 @llvm.ppc.altivec.vcmpeqfp.p(i32, <4 x float>, <4 x float>)
