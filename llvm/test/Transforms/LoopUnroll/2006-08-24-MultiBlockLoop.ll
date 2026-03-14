; RUN: opt < %s -passes=loop-unroll -S | grep bb72.2

define void @vorbis_encode_noisebias_setup() {
entry:
	br label %cond_true.outer
cond_true.outer:		; preds = %bb72, %entry
	%indvar1.ph = phi i32 [ 0, %entry ], [ %indvar.next2, %bb72 ]		; <i32> [#uses=1]
	br label %bb72
bb72:		; preds = %cond_true.outer
	%indvar.next2 = add i32 %indvar1.ph, 1		; <i32> [#uses=2]
	%exitcond3 = icmp eq i32 %indvar.next2, 3		; <i1> [#uses=1]
	br i1 %exitcond3, label %cond_true138, label %cond_true.outer
cond_true138:		; preds = %bb72
	ret void
}

