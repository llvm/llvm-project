; RUN: llvm-as < %s | opt -licm -loop-unswitch -disable-output
@g_56 = external global i16		; <i16*> [#uses=2]

define i32 @func_67(i32 %p_68, i8 signext  %p_69, i8 signext  %p_71) nounwind  {
entry:
	br label %bb
bb:		; preds = %bb44, %entry
	br label %bb3
bb3:		; preds = %bb36, %bb
	%bothcond = or i1 false, false		; <i1> [#uses=1]
	br i1 %bothcond, label %bb29, label %bb19
bb19:		; preds = %bb3
	br i1 false, label %bb36, label %bb29
bb29:		; preds = %bb19, %bb3
	ret i32 0
bb36:		; preds = %bb19
	store i16 0, i16* @g_56, align 2
	br i1 false, label %bb44, label %bb3
bb44:		; preds = %bb44, %bb36
	%tmp46 = load i16* @g_56, align 2		; <i16> [#uses=0]
	br i1 false, label %bb, label %bb44
}
