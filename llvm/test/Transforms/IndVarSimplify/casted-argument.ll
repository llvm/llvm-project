; RUN: opt < %s -passes=indvars -disable-output
; PR4009
; PR4038

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:32:32"
target triple = "i386-pc-linux-gnu"

define void @safe_bcopy(ptr %to) nounwind {
entry:
	%cmp11 = icmp ult ptr %to, null		; <i1> [#uses=1]
	br i1 %cmp11, label %loop, label %return

return:		; preds = %entry
	ret void

loop:		; preds = %loop, %if.else
	%pn = phi ptr [ %ge, %loop ], [ null, %entry ]		; <ptr> [#uses=1]
	%cp = ptrtoint ptr %to to i32		; <i32> [#uses=1]
	%su = sub i32 0, %cp		; <i32> [#uses=1]
	%ge = getelementptr i8, ptr %pn, i32 %su		; <ptr> [#uses=2]
	tail call void @bcopy(ptr %ge) nounwind
	br label %loop
}

define void @safe_bcopy_4038(ptr %from, ptr %to, i32 %size) nounwind {
entry:
	br i1 false, label %if.else, label %if.then12

if.then12:		; preds = %entry
	ret void

if.else:		; preds = %entry
	%sub.ptr.rhs.cast40 = ptrtoint ptr %from to i32		; <i32> [#uses=1]
	br label %if.end54

if.end54:		; preds = %if.end54, %if.else
	%sub.ptr4912.pn = phi ptr [ %sub.ptr4912, %if.end54 ], [ null, %if.else ]		; <ptr> [#uses=1]
	%sub.ptr7 = phi ptr [ %sub.ptr, %if.end54 ], [ null, %if.else ]		; <ptr> [#uses=2]
	%sub.ptr.rhs.cast46.pn = ptrtoint ptr %from to i32		; <i32> [#uses=1]
	%sub.ptr.lhs.cast45.pn = ptrtoint ptr %to to i32		; <i32> [#uses=1]
	%sub.ptr.sub47.pn = sub i32 %sub.ptr.rhs.cast46.pn, %sub.ptr.lhs.cast45.pn		; <i32> [#uses=1]
	%sub.ptr4912 = getelementptr i8, ptr %sub.ptr4912.pn, i32 %sub.ptr.sub47.pn		; <ptr> [#uses=2]
	tail call void @bcopy_4038(ptr %sub.ptr4912, ptr %sub.ptr7, i32 0) nounwind
	%sub.ptr = getelementptr i8, ptr %sub.ptr7, i32 %sub.ptr.rhs.cast40		; <ptr> [#uses=1]
	br label %if.end54
}

declare void @bcopy(ptr nocapture) nounwind

declare void @bcopy_4038(ptr, ptr, i32) nounwind
