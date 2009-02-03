; RUN: llvm-as < %s | llc -march=x86 -mattr=+sse42 -disable-mmx -o %t -f
; RUN: grep mulps  %t | count 1
; RUN: grep addps  %t | count 1

; widen a v3f32 to vfi32 to do a vector multiple and an add
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"

define void @update(<3 x float>* %dst, <3 x float>* %src, i32 %n) nounwind {
entry:
	%dst.addr = alloca <3 x float>*		; <<3 x float>**> [#uses=2]
	%src.addr = alloca <3 x float>*		; <<3 x float>**> [#uses=2]
	%n.addr = alloca i32		; <i32*> [#uses=2]
	%v = alloca <3 x float>, align 16		; <<3 x float>*> [#uses=2]
	%i = alloca i32, align 4		; <i32*> [#uses=6]
	store <3 x float>* %dst, <3 x float>** %dst.addr
	store <3 x float>* %src, <3 x float>** %src.addr
	store i32 %n, i32* %n.addr
	store <3 x float> < float 1.000000e+00, float 2.000000e+00, float 3.000000e+00 >, <3 x float>* %v
	store i32 0, i32* %i
	br label %forcond

forcond:		; preds = %forinc, %entry
	%tmp = load i32* %i		; <i32> [#uses=1]
	%tmp1 = load i32* %n.addr		; <i32> [#uses=1]
	%cmp = icmp slt i32 %tmp, %tmp1		; <i1> [#uses=1]
	br i1 %cmp, label %forbody, label %afterfor

forbody:		; preds = %forcond
	%tmp2 = load i32* %i		; <i32> [#uses=1]
	%tmp3 = load <3 x float>** %dst.addr		; <<3 x float>*> [#uses=1]
	%arrayidx = getelementptr <3 x float>* %tmp3, i32 %tmp2		; <<3 x float>*> [#uses=1]
	%tmp4 = load i32* %i		; <i32> [#uses=1]
	%tmp5 = load <3 x float>** %src.addr		; <<3 x float>*> [#uses=1]
	%arrayidx6 = getelementptr <3 x float>* %tmp5, i32 %tmp4		; <<3 x float>*> [#uses=1]
	%tmp7 = load <3 x float>* %arrayidx6		; <<3 x float>> [#uses=1]
	%tmp8 = load <3 x float>* %v		; <<3 x float>> [#uses=1]
	%mul = mul <3 x float> %tmp7, %tmp8		; <<3 x float>> [#uses=1]
	%add = add <3 x float> %mul, < float 0x409EE02900000000, float 0x409EE02900000000, float 0x409EE02900000000 >		; <<3 x float>> [#uses=1]
	store <3 x float> %add, <3 x float>* %arrayidx
	br label %forinc

forinc:		; preds = %forbody
	%tmp9 = load i32* %i		; <i32> [#uses=1]
	%inc = add i32 %tmp9, 1		; <i32> [#uses=1]
	store i32 %inc, i32* %i
	br label %forcond

afterfor:		; preds = %forcond
	ret void
}
