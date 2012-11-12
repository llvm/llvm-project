; RUN: llc < %s -relocation-model=pic | grep atoi | grep PLT
; PR4253
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"
target triple = "x86_64-unknown-linux-gnu"

define i32 @foo(i8* %x) nounwind readonly {
entry:
	%call = tail call fastcc i32 @atoi(i8* %x) nounwind readonly		; <i32> [#uses=1]
	ret i32 %call
}

define available_externally fastcc i32 @atoi(i8* %__nptr) nounwind readonly {
entry:
	%call = tail call i64 @strtol(i8* nocapture %__nptr, i8** null, i32 10) nounwind readonly		; <i64> [#uses=1]
	%conv = trunc i64 %call to i32		; <i32> [#uses=1]
	ret i32 %conv
}

declare i64 @strtol(i8*, i8** nocapture, i32) nounwind
