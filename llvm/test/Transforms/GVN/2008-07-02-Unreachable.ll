; RUN: opt < %s -passes=gvn -S | FileCheck %s
; PR2503

@g_3 = external global i8		; <i8*> [#uses=2]

define i8 @func_1(i32 %x, i32 %y) nounwind  {
entry:
  %A = alloca i8
    %cmp = icmp eq i32 %x, %y
	br i1 %cmp, label %ifelse, label %ifthen

ifthen:		; preds = %entry
	br label %ifend

ifelse:		; preds = %entry
	%tmp3 = load i8, i8* @g_3		; <i8> [#uses=0]
        store i8 %tmp3, i8* %A
	br label %afterfor

forcond:		; preds = %forinc
	br i1 false, label %afterfor, label %forbody

forbody:		; preds = %forcond
	br label %forinc

forinc:		; preds = %forbody
	br label %forcond

afterfor:		; preds = %forcond, %forcond.thread
	%tmp10 = load i8, i8* @g_3		; <i8> [#uses=0]
	ret i8 %tmp10
; CHECK: ret i8 %tmp3

ifend:		; preds = %afterfor, %ifthen
	ret i8 0
}
