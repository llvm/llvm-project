; RUN: opt < %s -passes=indvars
; PR4315

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"
target triple = "x86_64-undermydesk-freebsd8.0"
	%struct.mbuf = type <{ ptr, ptr, i32, i8, i8, i8, i8 }>

define i32 @crash(ptr %m, i1 %arg) nounwind {
entry:
	br label %for.cond

for.cond:		; preds = %if.end, %entry
	%i.0 = phi i32 [ 0, %entry ], [ %inc, %if.end ]		; <i32> [#uses=3]
	%chksum.0 = phi i8 [ 0, %entry ], [ %conv3, %if.end ]		; <i8> [#uses=3]
	%cmp = icmp slt i32 %i.0, 1		; <i1> [#uses=1]
	br i1 %cmp, label %for.body, label %do.body

for.body:		; preds = %for.cond
	br i1 %arg, label %if.end, label %do.body

if.end:		; preds = %for.body
	%i.02 = trunc i32 %i.0 to i8		; <i8> [#uses=1]
	%conv3 = add i8 %chksum.0, %i.02		; <i8> [#uses=1]
	%inc = add i32 %i.0, 1		; <i32> [#uses=1]
	br label %for.cond

do.body:		; preds = %do.cond, %for.body, %for.cond
	%chksum.2 = phi i8 [ undef, %do.cond ], [ %chksum.0, %for.body ], [ %chksum.0, %for.cond ]		; <i8> [#uses=1]
	br i1 %arg, label %do.cond, label %bb.nph

bb.nph:		; preds = %do.body
	br label %while.body

while.body:		; preds = %while.body, %bb.nph
	%chksum.13 = phi i8 [ undef, %while.body ], [ %chksum.2, %bb.nph ]		; <i8> [#uses=0]
	br i1 %arg, label %do.cond, label %while.body

do.cond:		; preds = %while.body, %do.body
	br i1 false, label %do.end, label %do.body

do.end:		; preds = %do.cond
	ret i32 0
}
