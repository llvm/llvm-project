; RUN: opt < %s -passes=indvars -disable-output
; PR4009
; PR4038

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:32:32"
target triple = "i386-pc-linux-gnu"

define void @safe_bcopy(i8* %to) nounwind {
entry:
	%cmp11 = icmp ult i8* %to, null		; <i1> [#uses=1]
	br i1 %cmp11, label %loop, label %return

return:		; preds = %entry
	ret void

loop:		; preds = %loop, %if.else
	%pn = phi i8* [ %ge, %loop ], [ null, %entry ]		; <i8*> [#uses=1]
	%cp = ptrtoint i8* %to to i32		; <i32> [#uses=1]
	%su = sub i32 0, %cp		; <i32> [#uses=1]
	%ge = getelementptr i8, i8* %pn, i32 %su		; <i8*> [#uses=2]
	tail call void @bcopy(i8* %ge) nounwind
	br label %loop
}

define void @safe_bcopy_4038(i8* %from, i8* %to, i32 %size) nounwind {
entry:
	br i1 false, label %if.else, label %if.then12

if.then12:		; preds = %entry
	ret void

if.else:		; preds = %entry
	%sub.ptr.rhs.cast40 = ptrtoint i8* %from to i32		; <i32> [#uses=1]
	br label %if.end54

if.end54:		; preds = %if.end54, %if.else
	%sub.ptr4912.pn = phi i8* [ %sub.ptr4912, %if.end54 ], [ null, %if.else ]		; <i8*> [#uses=1]
	%sub.ptr7 = phi i8* [ %sub.ptr, %if.end54 ], [ null, %if.else ]		; <i8*> [#uses=2]
	%sub.ptr.rhs.cast46.pn = ptrtoint i8* %from to i32		; <i32> [#uses=1]
	%sub.ptr.lhs.cast45.pn = ptrtoint i8* %to to i32		; <i32> [#uses=1]
	%sub.ptr.sub47.pn = sub i32 %sub.ptr.rhs.cast46.pn, %sub.ptr.lhs.cast45.pn		; <i32> [#uses=1]
	%sub.ptr4912 = getelementptr i8, i8* %sub.ptr4912.pn, i32 %sub.ptr.sub47.pn		; <i8*> [#uses=2]
	tail call void @bcopy_4038(i8* %sub.ptr4912, i8* %sub.ptr7, i32 0) nounwind
	%sub.ptr = getelementptr i8, i8* %sub.ptr7, i32 %sub.ptr.rhs.cast40		; <i8*> [#uses=1]
	br label %if.end54
}

declare void @bcopy(i8* nocapture) nounwind

declare void @bcopy_4038(i8*, i8*, i32) nounwind
