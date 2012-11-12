; RUN: llc -march=x86 -mcpu=generic -mattr=+sse42 < %s | FileCheck %s
; RUN: llc -march=x86 -mcpu=atom < %s | FileCheck -check-prefix=ATOM %s

; CHECK: paddd
; CHECK: movl
; CHECK: movlpd

; Scheduler causes produce a different instruction order
; ATOM: movl
; ATOM: paddd
; ATOM: movlpd

; bitcast a v4i16 to v2i32

define void @convert(<2 x i32>* %dst, <4 x i16>* %src) nounwind {
entry:
	%dst.addr = alloca <2 x i32>*		; <<2 x i32>**> [#uses=2]
	%src.addr = alloca <4 x i16>*		; <<4 x i16>**> [#uses=2]
	%i = alloca i32, align 4		; <i32*> [#uses=6]
	store <2 x i32>* %dst, <2 x i32>** %dst.addr
	store <4 x i16>* %src, <4 x i16>** %src.addr
	store i32 0, i32* %i
	br label %forcond

forcond:		; preds = %forinc, %entry
	%tmp = load i32* %i		; <i32> [#uses=1]
	%cmp = icmp slt i32 %tmp, 4		; <i1> [#uses=1]
	br i1 %cmp, label %forbody, label %afterfor

forbody:		; preds = %forcond
	%tmp1 = load i32* %i		; <i32> [#uses=1]
	%tmp2 = load <2 x i32>** %dst.addr		; <<2 x i32>*> [#uses=1]
	%arrayidx = getelementptr <2 x i32>* %tmp2, i32 %tmp1		; <<2 x i32>*> [#uses=1]
	%tmp3 = load i32* %i		; <i32> [#uses=1]
	%tmp4 = load <4 x i16>** %src.addr		; <<4 x i16>*> [#uses=1]
	%arrayidx5 = getelementptr <4 x i16>* %tmp4, i32 %tmp3		; <<4 x i16>*> [#uses=1]
	%tmp6 = load <4 x i16>* %arrayidx5		; <<4 x i16>> [#uses=1]
	%add = add <4 x i16> %tmp6, < i16 1, i16 1, i16 1, i16 1 >		; <<4 x i16>> [#uses=1]
	%conv = bitcast <4 x i16> %add to <2 x i32>		; <<2 x i32>> [#uses=1]
	store <2 x i32> %conv, <2 x i32>* %arrayidx
	br label %forinc

forinc:		; preds = %forbody
	%tmp7 = load i32* %i		; <i32> [#uses=1]
	%inc = add i32 %tmp7, 1		; <i32> [#uses=1]
	store i32 %inc, i32* %i
	br label %forcond

afterfor:		; preds = %forcond
	ret void
}
