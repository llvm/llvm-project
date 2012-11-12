; RUN: opt < %s -scalarrepl -loop-simplify -licm -disable-output -verify-dom-info -verify-loop-info

define void @inflate() {
entry:
	br label %loopentry.0.outer1111
loopentry.0.outer1111:		; preds = %then.41, %label.11, %loopentry.0.outer1111, %entry
	%left.0.ph1107 = phi i32 [ %tmp.1172, %then.41 ], [ 0, %entry ], [ %tmp.1172, %label.11 ], [ %left.0.ph1107, %loopentry.0.outer1111 ]		; <i32> [#uses=2]
	%tmp.1172 = sub i32 %left.0.ph1107, 0		; <i32> [#uses=2]
	switch i32 0, label %label.11 [
		 i32 23, label %loopentry.0.outer1111
		 i32 13, label %then.41
	]
label.11:		; preds = %loopentry.0.outer1111
	br label %loopentry.0.outer1111
then.41:		; preds = %loopentry.0.outer1111
	br label %loopentry.0.outer1111
}

