; RUN: opt < %s -passes=indvars
; PR4271

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"
target triple = "i386-apple-darwin10.0"
	%struct.xyz = type <{ i64, i64, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i64, [8 x i8], i64, i64, i32, i32, [4 x i32], i32, i32, i32, i32, i32, i32, [76 x i32], i32, [2 x %struct.uvw] }>
	%struct.uvw = type <{ i64, i64 }>

define i32 @foo(%struct.xyz* %header, i8* %p2, i8* %p3, i8* nocapture %p4) nounwind {
entry:
	br label %while.body.i

while.body.i:		; preds = %while.body.i, %entry
	br i1 undef, label %while.body.i, label %bcopy_internal.exit

bcopy_internal.exit:		; preds = %while.body.i
	%conv135 = ptrtoint %struct.xyz* %header to i32		; <i32> [#uses=1]
	%shr136 = lshr i32 %conv135, 12		; <i32> [#uses=1]
	br label %for.body

for.body:		; preds = %for.body, %bcopy_internal.exit
	%ppnum.052 = phi i32 [ %inc, %for.body ], [ %shr136, %bcopy_internal.exit ]		; <i32> [#uses=1]
	%inc = add i32 %ppnum.052, 1		; <i32> [#uses=2]
	%cmp = icmp ugt i32 %inc, undef		; <i1> [#uses=1]
	br i1 %cmp, label %if.then199, label %for.body

if.then199:		; preds = %if.then199, %for.body
	br label %if.then199
}

define i32 @same_thing_but_signed(%struct.xyz* %header, i8* %p2, i8* %p3, i8* nocapture %p4) nounwind {
entry:
	br label %while.body.i

while.body.i:		; preds = %while.body.i, %entry
	br i1 undef, label %while.body.i, label %bcopy_internal.exit

bcopy_internal.exit:		; preds = %while.body.i
	%conv135 = ptrtoint %struct.xyz* %header to i32		; <i32> [#uses=1]
	%shr136 = ashr i32 %conv135, 12		; <i32> [#uses=1]
	br label %for.body

for.body:		; preds = %for.body, %bcopy_internal.exit
	%ppnum.052 = phi i32 [ %inc, %for.body ], [ %shr136, %bcopy_internal.exit ]		; <i32> [#uses=1]
	%inc = add i32 %ppnum.052, 1		; <i32> [#uses=2]
	%cmp = icmp ugt i32 %inc, undef		; <i1> [#uses=1]
	br i1 %cmp, label %if.then199, label %for.body

if.then199:		; preds = %if.then199, %for.body
	br label %if.then199
}

define i32 @same_thing_but_multiplied(%struct.xyz* %header, i8* %p2, i8* %p3, i8* nocapture %p4) nounwind {
entry:
	br label %while.body.i

while.body.i:		; preds = %while.body.i, %entry
	br i1 undef, label %while.body.i, label %bcopy_internal.exit

bcopy_internal.exit:		; preds = %while.body.i
	%conv135 = ptrtoint %struct.xyz* %header to i32		; <i32> [#uses=1]
	%shr136 = shl i32 %conv135, 12		; <i32> [#uses=1]
	br label %for.body

for.body:		; preds = %for.body, %bcopy_internal.exit
	%ppnum.052 = phi i32 [ %inc, %for.body ], [ %shr136, %bcopy_internal.exit ]		; <i32> [#uses=1]
	%inc = add i32 %ppnum.052, 1		; <i32> [#uses=2]
	%cmp = icmp ugt i32 %inc, undef		; <i1> [#uses=1]
	br i1 %cmp, label %if.then199, label %for.body

if.then199:		; preds = %if.then199, %for.body
	br label %if.then199
}

define i32 @same_thing_but_xored(%struct.xyz* %header, i8* %p2, i8* %p3, i8* nocapture %p4) nounwind {
entry:
	br label %while.body.i

while.body.i:		; preds = %while.body.i, %entry
	br i1 undef, label %while.body.i, label %bcopy_internal.exit

bcopy_internal.exit:		; preds = %while.body.i
	%conv135 = ptrtoint %struct.xyz* %header to i32		; <i32> [#uses=1]
	%shr136 = xor i32 %conv135, 12		; <i32> [#uses=1]
	br label %for.body

for.body:		; preds = %for.body, %bcopy_internal.exit
	%ppnum.052 = phi i32 [ %inc, %for.body ], [ %shr136, %bcopy_internal.exit ]		; <i32> [#uses=1]
	%inc = add i32 %ppnum.052, 1		; <i32> [#uses=2]
	%cmp = icmp ugt i32 %inc, undef		; <i1> [#uses=1]
	br i1 %cmp, label %if.then199, label %for.body

if.then199:		; preds = %if.then199, %for.body
	br label %if.then199
}
