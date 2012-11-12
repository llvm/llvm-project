; RUN: llc < %s -march=x86-64 -mattr=+sse42  | FileCheck %s
; CHECK: movdqa
; CHECK: pmulld
; CHECK: psubd

; widen a v3i32 to v4i32 to do a vector multiple and a subtraction

define void @update(<3 x i32>* %dst, <3 x i32>* %src, i32 %n) nounwind {
entry:
	%dst.addr = alloca <3 x i32>*		; <<3 x i32>**> [#uses=2]
	%src.addr = alloca <3 x i32>*		; <<3 x i32>**> [#uses=2]
	%n.addr = alloca i32		; <i32*> [#uses=2]
	%v = alloca <3 x i32>, align 16		; <<3 x i32>*> [#uses=1]
	%i = alloca i32, align 4		; <i32*> [#uses=6]
	store <3 x i32>* %dst, <3 x i32>** %dst.addr
	store <3 x i32>* %src, <3 x i32>** %src.addr
	store i32 %n, i32* %n.addr
	store <3 x i32> < i32 1, i32 1, i32 1 >, <3 x i32>* %v
	store i32 0, i32* %i
	br label %forcond

forcond:		; preds = %forinc, %entry
	%tmp = load i32* %i		; <i32> [#uses=1]
	%tmp1 = load i32* %n.addr		; <i32> [#uses=1]
	%cmp = icmp slt i32 %tmp, %tmp1		; <i1> [#uses=1]
	br i1 %cmp, label %forbody, label %afterfor

forbody:		; preds = %forcond
	%tmp2 = load i32* %i		; <i32> [#uses=1]
	%tmp3 = load <3 x i32>** %dst.addr		; <<3 x i32>*> [#uses=1]
	%arrayidx = getelementptr <3 x i32>* %tmp3, i32 %tmp2		; <<3 x i32>*> [#uses=1]
	%tmp4 = load i32* %i		; <i32> [#uses=1]
	%tmp5 = load <3 x i32>** %src.addr		; <<3 x i32>*> [#uses=1]
	%arrayidx6 = getelementptr <3 x i32>* %tmp5, i32 %tmp4		; <<3 x i32>*> [#uses=1]
	%tmp7 = load <3 x i32>* %arrayidx6		; <<3 x i32>> [#uses=1]
	%mul = mul <3 x i32> %tmp7, < i32 4, i32 4, i32 4 >		; <<3 x i32>> [#uses=1]
	%sub = sub <3 x i32> %mul, < i32 3, i32 3, i32 3 >		; <<3 x i32>> [#uses=1]
	store <3 x i32> %sub, <3 x i32>* %arrayidx
	br label %forinc

forinc:		; preds = %forbody
	%tmp8 = load i32* %i		; <i32> [#uses=1]
	%inc = add i32 %tmp8, 1		; <i32> [#uses=1]
	store i32 %inc, i32* %i
	br label %forcond

afterfor:		; preds = %forcond
	ret void
}

