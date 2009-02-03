; RUN: llvm-as < %s | opt -predsimplify -verify

define void @dgefa() {
entry:
	br label %cond_true96
cond_true:		; preds = %cond_true96
	%tmp19 = icmp eq i32 %tmp10, %k.0		; <i1> [#uses=1]
	br i1 %tmp19, label %cond_next, label %cond_true20
cond_true20:		; preds = %cond_true
	br label %cond_next
cond_next:		; preds = %cond_true20, %cond_true
	%tmp84 = icmp sgt i32 %tmp3, 1999		; <i1> [#uses=0]
	ret void
cond_true96:		; preds = %cond_true96, %entry
	%k.0 = phi i32 [ 0, %entry ], [ 0, %cond_true96 ]		; <i32> [#uses=3]
	%tmp3 = add i32 %k.0, 1		; <i32> [#uses=1]
	%tmp10 = add i32 0, %k.0		; <i32> [#uses=1]
	br i1 false, label %cond_true96, label %cond_true
}

