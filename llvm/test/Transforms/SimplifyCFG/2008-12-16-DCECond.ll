; RUN: opt < %s -simplifycfg -S -hoist-common-insts=true | not grep icmp
; ModuleID = '/tmp/x.bc'
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"
target triple = "i686-pc-linux-gnu"

define i32 @x(i32 %x) {
entry:
	%cmp = icmp eq i32 %x, 8		; <i1> [#uses=1]
	br i1 %cmp, label %ifthen, label %ifend

ifthen:		; preds = %entry
	%call = call i32 (...) @foo()		; <i32> [#uses=0]
	br label %ifend

ifend:		; preds = %ifthen, %entry
	%cmp2 = icmp ne i32 %x, 8		; <i1> [#uses=1]
	br i1 %cmp2, label %ifthen3, label %ifend5

ifthen3:		; preds = %ifend
	%call4 = call i32 (...) @foo()		; <i32> [#uses=0]
	br label %ifend5

ifend5:		; preds = %ifthen3, %ifend
	%cmp7 = icmp eq i32 %x, 9		; <i1> [#uses=1]
	br i1 %cmp7, label %ifthen8, label %ifend10

ifthen8:		; preds = %ifend5
	%call9 = call i32 (...) @bar()		; <i32> [#uses=0]
	br label %ifend10

ifend10:		; preds = %ifthen8, %ifend5
	%cmp12 = icmp ne i32 %x, 9		; <i1> [#uses=1]
	br i1 %cmp12, label %ifthen13, label %ifend15

ifthen13:		; preds = %ifend10
	%call14 = call i32 (...) @bar()		; <i32> [#uses=0]
	br label %ifend15

ifend15:		; preds = %ifthen13, %ifend10
	ret i32 0
}

declare i32 @foo(...)

declare i32 @bar(...)

