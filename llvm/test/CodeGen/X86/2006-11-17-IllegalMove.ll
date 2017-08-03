; RUN: llc < %s -march=x86-64 > %t
; RUN: grep movb %t | count 1
; RUN: grep "movzb[wl]" %t


define void @handle_vector_size_attribute() nounwind {
entry:
	%tmp69 = load i32, i32* null		; <i32> [#uses=1]
	switch i32 %tmp69, label %bb84 [
		 i32 2, label %bb77
		 i32 1, label %bb77
	]

bb77:		; preds = %entry, %entry
	%tmp99 = udiv i64 0, 0		; <i64> [#uses=1]
	%tmp = load i8, i8* null		; <i8> [#uses=1]
	%tmp114 = icmp eq i64 0, 0		; <i1> [#uses=1]
	br label %cond_true115

bb84:		; preds = %entry
	ret void

cond_true115:		; preds = %bb77
	%tmp118 = load i8, i8* null		; <i8> [#uses=1]
	br label %cond_true120

cond_true120:		; preds = %cond_true115
	%tmp127 = udiv i8 %tmp, %tmp118		; <i8> [#uses=1]
	%tmp127.upgrd.1 = zext i8 %tmp127 to i64		; <i64> [#uses=1]
	br label %cond_next129

cond_next129:		; preds = %cond_true120, %cond_true115
	%iftmp.30.0 = phi i64 [ %tmp127.upgrd.1, %cond_true120 ]		; <i64> [#uses=1]
	%tmp132 = icmp eq i64 %iftmp.30.0, %tmp99		; <i1> [#uses=1]
	br i1 %tmp132, label %cond_false148, label %cond_next136

cond_next136:		; preds = %cond_next129, %bb77
	ret void

cond_false148:		; preds = %cond_next129
	ret void
}
