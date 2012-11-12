; RUN: llc < %s -march=x86 | grep mov | count 3

define fastcc i32 @_Z18yy_get_next_bufferv() nounwind {
entry:
	br label %bb131

bb116:		; preds = %bb131
	%tmp125126 = trunc i32 %c.1 to i8		; <i8> [#uses=1]
	store i8 %tmp125126, i8* null, align 1
	br label %bb131

bb131:		; preds = %bb116, %entry
	%c.2 = phi i32 [ %c.1, %bb116 ], [ 42, %entry ]		; <i32> [#uses=1]
	%c.1 = select i1 false, i32 0, i32 %c.2		; <i32> [#uses=4]
	%tmp181 = icmp eq i32 %c.1, -1		; <i1> [#uses=1]
	br i1 %tmp181, label %bb158, label %bb116

bb158:		; preds = %bb131
	br i1 true, label %cond_true163, label %cond_next178

cond_true163:		; preds = %bb158
	%tmp172173 = trunc i32 %c.1 to i8		; <i8> [#uses=1]
	store i8 %tmp172173, i8* null, align 1
	br label %cond_next178

cond_next178:		; preds = %cond_true163, %bb158
	%tmp180 = icmp eq i32 %c.1, -1		; <i1> [#uses=1]
	br i1 %tmp180, label %cond_next184, label %cond_next199

cond_next184:		; preds = %cond_next178
	ret i32 0

cond_next199:		; preds = %cond_next178
	ret i32 0
}
