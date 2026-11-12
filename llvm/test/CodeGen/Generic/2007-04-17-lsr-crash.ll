; RUN: llc < %s

define void @foo(i32 %inTextSize) {
entry:
	br label %bb236.outer

cond_next193:		; preds = %bb236
	%tmp211 = add i32 %inTextSize_addr.1.ph17, -2		; <i32> [#uses=1]
	br i1 false, label %cond_next232, label %cond_true227

cond_true227:		; preds = %cond_next193
	ret void

cond_next232:		; preds = %cond_next193
	%indvar.next49 = add i32 %indvar48, 1		; <i32> [#uses=1]
	br label %bb236.outer

bb236.outer:		; preds = %cond_next232, %entry
	%indvar48 = phi i32 [ %indvar.next49, %cond_next232 ], [ 0, %entry ]		; <i32> [#uses=2]
	%inTextSize_addr.1.ph17 = phi i32 [ %tmp211, %cond_next232 ], [ %inTextSize, %entry ]		; <i32> [#uses=3]
	%tmp.50 = sub i32 0, %indvar48		; <i32> [#uses=1]
	%tmp219 = icmp eq i32 %tmp.50, 0		; <i1> [#uses=1]
	br i1 %tmp219, label %bb236.us, label %bb236

bb236.us:		; preds = %bb236.outer
	%inTextSize_addr.1.us = add i32 0, %inTextSize_addr.1.ph17		; <i32> [#uses=0]
	ret void

bb236:		; preds = %bb236.outer
	%tmp238 = icmp eq i32 %inTextSize_addr.1.ph17, 0		; <i1> [#uses=1]
	br i1 %tmp238, label %exit, label %cond_next193

exit:		; preds = %bb236
	ret void
}
