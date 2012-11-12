; RUN: llc < %s -march=x86-64 -mattr=+sse42 | FileCheck %s
; CHECK: psubw
; CHECK-NEXT: pmullw

; Widen a v5i16 to v8i16 to do a vector sub and multiple

define void @update(<5 x i16>* %dst, <5 x i16>* %src, i32 %n) nounwind {
entry:
	%dst.addr = alloca <5 x i16>*		; <<5 x i16>**> [#uses=2]
	%src.addr = alloca <5 x i16>*		; <<5 x i16>**> [#uses=2]
	%n.addr = alloca i32		; <i32*> [#uses=2]
	%v = alloca <5 x i16>, align 16		; <<5 x i16>*> [#uses=1]
	%i = alloca i32, align 4		; <i32*> [#uses=6]
	store <5 x i16>* %dst, <5 x i16>** %dst.addr
	store <5 x i16>* %src, <5 x i16>** %src.addr
	store i32 %n, i32* %n.addr
	store <5 x i16> < i16 1, i16 1, i16 1, i16 0, i16 0 >, <5 x i16>* %v
	store i32 0, i32* %i
	br label %forcond

forcond:		; preds = %forinc, %entry
	%tmp = load i32* %i		; <i32> [#uses=1]
	%tmp1 = load i32* %n.addr		; <i32> [#uses=1]
	%cmp = icmp slt i32 %tmp, %tmp1		; <i1> [#uses=1]
	br i1 %cmp, label %forbody, label %afterfor

forbody:		; preds = %forcond
	%tmp2 = load i32* %i		; <i32> [#uses=1]
	%tmp3 = load <5 x i16>** %dst.addr		; <<5 x i16>*> [#uses=1]
	%arrayidx = getelementptr <5 x i16>* %tmp3, i32 %tmp2		; <<5 x i16>*> [#uses=1]
	%tmp4 = load i32* %i		; <i32> [#uses=1]
	%tmp5 = load <5 x i16>** %src.addr		; <<5 x i16>*> [#uses=1]
	%arrayidx6 = getelementptr <5 x i16>* %tmp5, i32 %tmp4		; <<5 x i16>*> [#uses=1]
	%tmp7 = load <5 x i16>* %arrayidx6		; <<5 x i16>> [#uses=1]
	%sub = sub <5 x i16> %tmp7, < i16 271, i16 271, i16 271, i16 271, i16 271 >		; <<5 x i16>> [#uses=1]
	%mul = mul <5 x i16> %sub, < i16 2, i16 2, i16 2, i16 2, i16 2 >		; <<5 x i16>> [#uses=1]
	store <5 x i16> %mul, <5 x i16>* %arrayidx
	br label %forinc

forinc:		; preds = %forbody
	%tmp8 = load i32* %i		; <i32> [#uses=1]
	%inc = add i32 %tmp8, 1		; <i32> [#uses=1]
	store i32 %inc, i32* %i
	br label %forcond

afterfor:		; preds = %forcond
	ret void
}

