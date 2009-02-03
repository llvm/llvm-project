; RUN: llvm-as < %s | opt -loop-rotate -disable-output

define void @findAllPairs() {
entry:
	br i1 false, label %bb139, label %cond_true
cond_true:		; preds = %entry
	ret void
bb90:		; preds = %bb139
	br i1 false, label %bb136, label %cond_next121
cond_next121:		; preds = %bb90
	br i1 false, label %bb136, label %bb127
bb127:		; preds = %cond_next121
	br label %bb136
bb136:		; preds = %bb127, %cond_next121, %bb90
	%changes.1 = phi i32 [ %changes.2, %bb90 ], [ %changes.2, %cond_next121 ], [ 1, %bb127 ]		; <i32> [#uses=1]
	br label %bb139
bb139:		; preds = %bb136, %entry
	%changes.2 = phi i32 [ %changes.1, %bb136 ], [ 0, %entry ]		; <i32> [#uses=3]
	br i1 false, label %bb90, label %bb142
bb142:		; preds = %bb139
	%changes.2.lcssa = phi i32 [ %changes.2, %bb139 ]		; <i32> [#uses=0]
	ret void
}

