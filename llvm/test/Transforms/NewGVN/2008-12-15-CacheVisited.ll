; RUN: opt < %s -newgvn | llvm-dis
; Cached results must be added to and verified against the visited sets.
; PR3217

define fastcc void @gen_field_die(i32* %decl) nounwind {
entry:
	br i1 false, label %bb203, label %bb202

bb202:		; preds = %entry
	unreachable

bb203:		; preds = %entry
	%tmp = getelementptr i32, i32* %decl, i32 1		; <i32*> [#uses=1]
	%tmp1 = load i32, i32* %tmp, align 4		; <i32> [#uses=0]
	br i1 false, label %bb207, label %bb204

bb204:		; preds = %bb203
	%tmp2 = getelementptr i32, i32* %decl, i32 1		; <i32*> [#uses=1]
	br label %bb208

bb207:		; preds = %bb203
	br label %bb208

bb208:		; preds = %bb207, %bb204
	%iftmp.1374.0.in = phi i32* [ null, %bb207 ], [ %tmp2, %bb204 ]		; <i32*> [#uses=1]
	%iftmp.1374.0 = load i32, i32* %iftmp.1374.0.in		; <i32> [#uses=0]
	unreachable
}
