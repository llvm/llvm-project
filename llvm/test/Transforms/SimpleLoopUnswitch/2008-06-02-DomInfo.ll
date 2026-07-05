; RUN: opt < %s -passes='loop-mssa(simple-loop-unswitch),instcombine,gvn' -verify-memoryssa -disable-output
; PR2372
target triple = "i386-pc-linux-gnu"

define i32 @func_3(i16 signext  %p_5, i16 signext  %p_6) nounwind  {
entry:
	%tmp3 = icmp eq i16 %p_5, 0		; <i1> [#uses=1]
	%tmp1314 = sext i16 %p_6 to i32		; <i32> [#uses=1]
	%tmp28 = icmp ugt i32 %tmp1314, 3		; <i1> [#uses=1]
	%bothcond = or i1 %tmp28, false		; <i1> [#uses=1]
	br label %bb
bb:		; preds = %bb54, %entry
	br i1 %tmp3, label %bb54, label %bb5
bb5:		; preds = %bb
	br i1 %bothcond, label %bb54, label %bb31
bb31:		; preds = %bb5
	br label %bb54
bb54:		; preds = %bb31, %bb5, %bb
	br i1 false, label %bb64, label %bb
bb64:		; preds = %bb54
	%tmp6566 = sext i16 %p_6 to i32		; <i32> [#uses=1]
	%tmp68 = tail call i32 (...) @func_18( i32 1, i32 %tmp6566, i32 1 ) nounwind 		; <i32> [#uses=0]
	ret i32 undef
}

declare i32 @func_18(...)
