; Split loop. Save last value. Split value is off by one in this example.
; RUN: llvm-as < %s | opt -loop-index-split -disable-output -stats |& \
; RUN: grep "loop-index-split" | count 1

@k = external global i32		; <i32*> [#uses=2]

define void @foobar(i32 %a, i32 %b) {
entry:
	br label %bb

bb:		; preds = %cond_next16, %entry
	%i.01.0 = phi i32 [ 0, %entry ], [ %tmp18, %cond_next16 ]		; <i32> [#uses=5]
	%tsum.18.0 = phi i32 [ 42, %entry ], [ %tsum.013.1, %cond_next16 ]		; <i32> [#uses=3]
	%tmp1 = icmp sgt i32 %i.01.0, 50		; <i1> [#uses=1]
	br i1 %tmp1, label %cond_true, label %cond_false

cond_true:		; preds = %bb
	%tmp4 = tail call i32 @foo( i32 %i.01.0 )		; <i32> [#uses=1]
	%tmp6 = add i32 %tmp4, %tsum.18.0		; <i32> [#uses=2]
	%tmp914 = load i32* @k, align 4		; <i32> [#uses=1]
	%tmp1015 = icmp eq i32 %tmp914, 0		; <i1> [#uses=1]
	br i1 %tmp1015, label %cond_next16, label %cond_true13

cond_false:		; preds = %bb
	%tmp8 = tail call i32 @bar( i32 %i.01.0 )		; <i32> [#uses=0]
	%tmp9 = load i32* @k, align 4		; <i32> [#uses=1]
	%tmp10 = icmp eq i32 %tmp9, 0		; <i1> [#uses=1]
	br i1 %tmp10, label %cond_next16, label %cond_true13

cond_true13:		; preds = %cond_false, %cond_true
	%tsum.013.0 = phi i32 [ %tmp6, %cond_true ], [ %tsum.18.0, %cond_false ]		; <i32> [#uses=1]
	%tmp15 = tail call i32 @bar( i32 %i.01.0 )		; <i32> [#uses=0]
	br label %cond_next16

cond_next16:		; preds = %cond_false, %cond_true, %cond_true13
	%tsum.013.1 = phi i32 [ %tsum.013.0, %cond_true13 ], [ %tmp6, %cond_true ], [ %tsum.18.0, %cond_false ]		; <i32> [#uses=2]
	%tmp18 = add i32 %i.01.0, 1		; <i32> [#uses=3]
	%tmp21 = icmp slt i32 %tmp18, 100		; <i1> [#uses=1]
	br i1 %tmp21, label %bb, label %bb24

bb24:		; preds = %cond_next16
	%tmp18.lcssa = phi i32 [ %tmp18, %cond_next16 ]		; <i32> [#uses=1]
	%tsum.013.1.lcssa = phi i32 [ %tsum.013.1, %cond_next16 ]		; <i32> [#uses=1]
	%tmp27 = tail call i32 @t( i32 %tmp18.lcssa, i32 %tsum.013.1.lcssa )		; <i32> [#uses=0]
	ret void
}

declare i32 @foo(i32)

declare i32 @bar(i32)

declare i32 @t(i32, i32)
