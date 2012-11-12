; RUN: llc < %s -mcpu=generic -march=x86 -mattr=+sse42 -post-RA-scheduler=true | FileCheck %s
; CHECK: incl
; CHECK: incl
; CHECK: incl

; Widen a v3i16 to v8i16 to do a vector add

@.str = internal constant [4 x i8] c"%d \00"		; <[4 x i8]*> [#uses=1]
@.str1 = internal constant [2 x i8] c"\0A\00"		; <[2 x i8]*> [#uses=1]

define void @update(<3 x i16>* %dst, <3 x i16>* %src, i32 %n) nounwind {
entry:
	%dst.addr = alloca <3 x i16>*		; <<3 x i16>**> [#uses=2]
	%src.addr = alloca <3 x i16>*		; <<3 x i16>**> [#uses=2]
	%n.addr = alloca i32		; <i32*> [#uses=2]
	%v = alloca <3 x i16>, align 8		; <<3 x i16>*> [#uses=1]
	%i = alloca i32, align 4		; <i32*> [#uses=6]
	store <3 x i16>* %dst, <3 x i16>** %dst.addr
	store <3 x i16>* %src, <3 x i16>** %src.addr
	store i32 %n, i32* %n.addr
	store <3 x i16> < i16 1, i16 1, i16 1 >, <3 x i16>* %v
	store i32 0, i32* %i
	br label %forcond

forcond:		; preds = %forinc, %entry
	%tmp = load i32* %i		; <i32> [#uses=1]
	%tmp1 = load i32* %n.addr		; <i32> [#uses=1]
	%cmp = icmp slt i32 %tmp, %tmp1		; <i1> [#uses=1]
	br i1 %cmp, label %forbody, label %afterfor

forbody:		; preds = %forcond
	%tmp2 = load i32* %i		; <i32> [#uses=1]
	%tmp3 = load <3 x i16>** %dst.addr		; <<3 x i16>*> [#uses=1]
	%arrayidx = getelementptr <3 x i16>* %tmp3, i32 %tmp2		; <<3 x i16>*> [#uses=1]
	%tmp4 = load i32* %i		; <i32> [#uses=1]
	%tmp5 = load <3 x i16>** %src.addr		; <<3 x i16>*> [#uses=1]
	%arrayidx6 = getelementptr <3 x i16>* %tmp5, i32 %tmp4		; <<3 x i16>*> [#uses=1]
	%tmp7 = load <3 x i16>* %arrayidx6		; <<3 x i16>> [#uses=1]
	%add = add <3 x i16> %tmp7, < i16 1, i16 1, i16 1 >		; <<3 x i16>> [#uses=1]
	store <3 x i16> %add, <3 x i16>* %arrayidx
	br label %forinc

forinc:		; preds = %forbody
	%tmp8 = load i32* %i		; <i32> [#uses=1]
	%inc = add i32 %tmp8, 1		; <i32> [#uses=1]
	store i32 %inc, i32* %i
	br label %forcond

afterfor:		; preds = %forcond
	ret void
}

