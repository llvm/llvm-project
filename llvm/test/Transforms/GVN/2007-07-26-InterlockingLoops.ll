; RUN: llvm-as < %s | opt -gvn | llvm-dis | grep {tmp17625.* = phi i32. }
; RUN: llvm-as < %s | opt -gvn | llvm-dis | grep {tmp17631.* = phi i32. }

@last = external global [65 x i32*]		; <[65 x i32*]*> [#uses=1]

define i32 @NextRootMove(i32 %wtm) {
cond_next95:		; preds = %cond_true85, %cond_true79, %cond_true73, %bb68
	%tmp17618 = load i32** getelementptr ([65 x i32*]* @last, i32 0, i32 1), align 4		; <i32*> [#uses=0]
	br label %cond_true116

cond_true116:		; preds = %cond_true111
	br i1 false, label %cond_true128, label %cond_true145

cond_true128:		; preds = %cond_true121
	%tmp17625 = load i32** getelementptr ([65 x i32*]* @last, i32 0, i32 1), align 4		; <i32*> [#uses=0]
	br i1 false, label %bb98.backedge, label %return.loopexit

bb98.backedge:		; preds = %bb171, %cond_true145, %cond_true128
	br label %cond_true116

cond_true145:		; preds = %cond_false
	%tmp17631 = load i32** getelementptr ([65 x i32*]* @last, i32 0, i32 1), align 4		; <i32*> [#uses=0]
	br i1 false, label %bb98.backedge, label %return.loopexit

return.loopexit:		; preds = %bb171, %cond_true145, %cond_true128
	br label %return

return:		; preds = %return.loopexit, %cond_next95, %cond_true85
	ret i32 0
}
