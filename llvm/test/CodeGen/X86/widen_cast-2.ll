; RUN: llvm-as < %s | llc -march=x86 -mattr=+sse42 -disable-mmx -o %t -f
; RUN: grep pextrd  %t | count 5
; RUN: grep movd  %t | count 3

; bitcast v14i16 to v7i32

define void @convert(<7 x i32>* %dst, <14 x i16>* %src) nounwind {
entry:
	%dst.addr = alloca <7 x i32>*		; <<7 x i32>**> [#uses=2]
	%src.addr = alloca <14 x i16>*		; <<14 x i16>**> [#uses=2]
	%i = alloca i32, align 4		; <i32*> [#uses=6]
	store <7 x i32>* %dst, <7 x i32>** %dst.addr
	store <14 x i16>* %src, <14 x i16>** %src.addr
	store i32 0, i32* %i
	br label %forcond

forcond:		; preds = %forinc, %entry
	%tmp = load i32* %i		; <i32> [#uses=1]
	%cmp = icmp slt i32 %tmp, 4		; <i1> [#uses=1]
	br i1 %cmp, label %forbody, label %afterfor

forbody:		; preds = %forcond
	%tmp1 = load i32* %i		; <i32> [#uses=1]
	%tmp2 = load <7 x i32>** %dst.addr		; <<2 x i32>*> [#uses=1]
	%arrayidx = getelementptr <7 x i32>* %tmp2, i32 %tmp1		; <<7 x i32>*> [#uses=1]
	%tmp3 = load i32* %i		; <i32> [#uses=1]
	%tmp4 = load <14 x i16>** %src.addr		; <<4 x i16>*> [#uses=1]
	%arrayidx5 = getelementptr <14 x i16>* %tmp4, i32 %tmp3		; <<4 x i16>*> [#uses=1]
	%tmp6 = load <14 x i16>* %arrayidx5		; <<4 x i16>> [#uses=1]
	%add = add <14 x i16> %tmp6, < i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1 >		; <<4 x i16>> [#uses=1]
	%conv = bitcast <14 x i16> %add to <7 x i32>		; <<7 x i32>> [#uses=1]
	store <7 x i32> %conv, <7 x i32>* %arrayidx
	br label %forinc

forinc:		; preds = %forbody
	%tmp7 = load i32* %i		; <i32> [#uses=1]
	%inc = add i32 %tmp7, 1		; <i32> [#uses=1]
	store i32 %inc, i32* %i
	br label %forcond

afterfor:		; preds = %forcond
	ret void
}
