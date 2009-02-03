; RUN: llvm-as < %s | opt -gvn | llvm-dis | grep undef
; PR2503

@g_3 = external global i8		; <i8*> [#uses=2]

define i8 @func_1() nounwind  {
entry:
	br i1 false, label %ifelse, label %ifthen

ifthen:		; preds = %entry
	br label %ifend

ifelse:		; preds = %entry
	%tmp3 = load i8* @g_3		; <i8> [#uses=0]
	br label %forcond.thread

forcond.thread:		; preds = %ifelse
	br label %afterfor

forcond:		; preds = %forinc
	br i1 false, label %afterfor, label %forbody

forbody:		; preds = %forcond
	br label %forinc

forinc:		; preds = %forbody
	br label %forcond

afterfor:		; preds = %forcond, %forcond.thread
	%tmp10 = load i8* @g_3		; <i8> [#uses=0]
	ret i8 %tmp10

ifend:		; preds = %afterfor, %ifthen
	ret i8 0
}
