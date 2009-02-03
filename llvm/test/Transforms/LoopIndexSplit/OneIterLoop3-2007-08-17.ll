; Loop is elimianted. Save last value assignments, including induction variable.
; RUN: llvm-as < %s | opt -loop-index-split -disable-output -stats | not grep "loop-index-split"

declare i32 @foo(i32)
declare i32 @bar(i32, i32)

define void @foobar(i32 %a, i32 %b) {
entry:
	br label %bb

bb:		; preds = %cond_next, %entry
	%i.01.0 = phi i32 [ 0, %entry ], [ %tmp8, %cond_next ]		; <i32> [#uses=3]
	%tsum.16.0 = phi i32 [ 42, %entry ], [ %tsum.0, %cond_next ]		; <i32> [#uses=2]
	%tmp1 = icmp eq i32 %i.01.0, 50		; <i1> [#uses=1]
	br i1 %tmp1, label %cond_true, label %cond_next

cond_true:		; preds = %bb
	%tmp4 = tail call i32 @foo( i32 %i.01.0 )		; <i32> [#uses=1]
	%tmp6 = add i32 %tmp4, %tsum.16.0		; <i32> [#uses=1]
	br label %cond_next

cond_next:		; preds = %bb, %cond_true
	%tsum.0 = phi i32 [ %tmp6, %cond_true ], [ %tsum.16.0, %bb ]		; <i32> [#uses=2]
	%tmp8 = add i32 %i.01.0, 1		; <i32> [#uses=3]
	%tmp11 = icmp slt i32 %tmp8, 100		; <i1> [#uses=1]
	br i1 %tmp11, label %bb, label %bb14

bb14:		; preds = %cond_next
	%tmp8.lcssa = phi i32 [ %tmp8, %cond_next ]		; <i32> [#uses=1]
	%tsum.0.lcssa = phi i32 [ %tsum.0, %cond_next ]		; <i32> [#uses=1]
	%tmp17 = tail call i32 @bar( i32 %tmp8.lcssa, i32 %tsum.0.lcssa )		; <i32> [#uses=0]
	ret void
}

