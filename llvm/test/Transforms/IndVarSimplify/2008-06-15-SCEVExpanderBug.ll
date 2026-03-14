; RUN: opt < %s -passes=indvars -disable-output
; PR2434

define fastcc void @regcppop() nounwind  {
entry:
	%tmp61 = add i32 0, -5		; <i32> [#uses=1]
	br label %bb

bb:		; preds = %bb, %entry
	%PL_savestack_ix.tmp.0 = phi i32 [ %tmp61, %entry ], [ %tmp127, %bb ]		; <i32> [#uses=2]
	%indvar10 = phi i32 [ 0, %entry ], [ %indvar.next11, %bb ]		; <i32> [#uses=2]
	%tmp13 = mul i32 %indvar10, -4		; <i32> [#uses=0]
	%tmp111 = add i32 %PL_savestack_ix.tmp.0, -3		; <i32> [#uses=0]
	%tmp127 = add i32 %PL_savestack_ix.tmp.0, -4		; <i32> [#uses=1]
	%indvar.next11 = add i32 %indvar10, 1		; <i32> [#uses=1]
	br label %bb
}
