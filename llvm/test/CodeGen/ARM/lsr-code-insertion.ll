; RUN: llc < %s | FileCheck %s
; This test really wants to check that the resultant "cond_true" block only 
; has a single store in it, and that cond_true55 only has code to materialize 
; the constant and do a store.  We do *not* want something like this:
;
;LBB1_3: @cond_true
;        add r8, r0, r6
;        str r10, [r8, #+4]
;
; CHECK: ldr [[R6:r[0-9*]+]], LCP
; CHECK: cmp {{.*}}, [[R6]]
; CHECK: ldrle
; CHECK-NEXT: strle

target triple = "arm-apple-darwin8"

define void @foo(i32* %mc, i32* %mpp, i32* %ip, i32* %dpp, i32* %tpmm, i32 %M, i32* %tpim, i32* %tpdm, i32* %bp, i32* %ms, i32 %xmb) {
entry:
	%tmp6584 = icmp slt i32 %M, 1		; <i1> [#uses=1]
	br i1 %tmp6584, label %return, label %bb

bb:		; preds = %cond_next59, %entry
	%indvar = phi i32 [ 0, %entry ], [ %k.069.0, %cond_next59 ]		; <i32> [#uses=6]
	%k.069.0 = add i32 %indvar, 1		; <i32> [#uses=3]
	%tmp3 = getelementptr i32* %mpp, i32 %indvar		; <i32*> [#uses=1]
	%tmp4 = load i32* %tmp3		; <i32> [#uses=1]
	%tmp8 = getelementptr i32* %tpmm, i32 %indvar		; <i32*> [#uses=1]
	%tmp9 = load i32* %tmp8		; <i32> [#uses=1]
	%tmp10 = add i32 %tmp9, %tmp4		; <i32> [#uses=2]
	%tmp13 = getelementptr i32* %mc, i32 %k.069.0		; <i32*> [#uses=5]
	store i32 %tmp10, i32* %tmp13
	%tmp17 = getelementptr i32* %ip, i32 %indvar		; <i32*> [#uses=1]
	%tmp18 = load i32* %tmp17		; <i32> [#uses=1]
	%tmp22 = getelementptr i32* %tpim, i32 %indvar		; <i32*> [#uses=1]
	%tmp23 = load i32* %tmp22		; <i32> [#uses=1]
	%tmp24 = add i32 %tmp23, %tmp18		; <i32> [#uses=2]
	%tmp30 = icmp sgt i32 %tmp24, %tmp10		; <i1> [#uses=1]
	br i1 %tmp30, label %cond_true, label %cond_next

cond_true:		; preds = %bb
	store i32 %tmp24, i32* %tmp13
	br label %cond_next

cond_next:		; preds = %cond_true, %bb
	%tmp39 = load i32* %tmp13		; <i32> [#uses=1]
	%tmp42 = getelementptr i32* %ms, i32 %k.069.0		; <i32*> [#uses=1]
	%tmp43 = load i32* %tmp42		; <i32> [#uses=1]
	%tmp44 = add i32 %tmp43, %tmp39		; <i32> [#uses=2]
	store i32 %tmp44, i32* %tmp13
	%tmp52 = icmp slt i32 %tmp44, -987654321		; <i1> [#uses=1]
	br i1 %tmp52, label %cond_true55, label %cond_next59

cond_true55:		; preds = %cond_next
	store i32 -987654321, i32* %tmp13
	br label %cond_next59

cond_next59:		; preds = %cond_true55, %cond_next
	%tmp61 = add i32 %indvar, 2		; <i32> [#uses=1]
	%tmp65 = icmp sgt i32 %tmp61, %M		; <i1> [#uses=1]
	br i1 %tmp65, label %return, label %bb

return:		; preds = %cond_next59, %entry
	ret void
}
