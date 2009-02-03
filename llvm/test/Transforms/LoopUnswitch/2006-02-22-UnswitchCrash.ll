; RUN: llvm-as < %s | opt -loop-unswitch -disable-output

define void @sort_Eq(i32* %S2) {
entry:
	br i1 false, label %list_Length.exit, label %cond_true.i
cond_true.i:		; preds = %entry
	ret void
list_Length.exit:		; preds = %entry
	br i1 false, label %list_Length.exit9, label %cond_true.i5
cond_true.i5:		; preds = %list_Length.exit
	ret void
list_Length.exit9:		; preds = %list_Length.exit
	br i1 false, label %bb78, label %return
bb44:		; preds = %bb78, %cond_next68
	br i1 %tmp49.not, label %bb62, label %bb62.loopexit
bb62.loopexit:		; preds = %bb44
	br label %bb62
bb62:		; preds = %bb62.loopexit, %bb44
	br i1 false, label %return.loopexit, label %cond_next68
cond_next68:		; preds = %bb62
	br i1 false, label %return.loopexit, label %bb44
bb78:		; preds = %list_Length.exit9
	%tmp49.not = icmp eq i32* %S2, null		; <i1> [#uses=1]
	br label %bb44
return.loopexit:		; preds = %cond_next68, %bb62
	%retval.0.ph = phi i32 [ 1, %cond_next68 ], [ 0, %bb62 ]		; <i32> [#uses=1]
	br label %return
return:		; preds = %return.loopexit, %list_Length.exit9
	%retval.0 = phi i32 [ 0, %list_Length.exit9 ], [ %retval.0.ph, %return.loopexit ]		; <i32> [#uses=0]
	ret void
}

