; RUN: llvm-as < %s | opt -gvnpre | llvm-dis

define fastcc void @compute_max_score_1() {
entry:
	%tmp7 = sub i32 0, 0		; <i32> [#uses=0]
	br label %bb

bb:		; preds = %bb212, %entry
	%indvar29 = phi i32 [ 0, %entry ], [ %indvar.next30, %bb212 ]		; <i32> [#uses=2]
	%j.01.0 = sub i32 %indvar29, 0		; <i32> [#uses=0]
	br label %cond_next166

cond_next166:		; preds = %cond_next166, %bb
	br i1 false, label %bb212, label %cond_next166

bb212:		; preds = %cond_next166
	%indvar.next30 = add i32 %indvar29, 1		; <i32> [#uses=1]
	br i1 false, label %return, label %bb

return:		; preds = %bb212
	ret void
}
