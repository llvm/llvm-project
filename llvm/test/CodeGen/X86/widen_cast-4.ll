; RUN: llvm-as < %s | llc -march=x86 -mattr=+sse42 -disable-mmx -o %t -f
; RUN: grep sarb  %t | count 16

; v8i8 that is widen to v16i8 then split
; FIXME: This is widen to v16i8 and split to 16 and we then rebuild the vector.
; Unfortunately, we don't split the store so we don't get the code we want.

define void @update(i64* %dst_i, i64* %src_i, i32 %n) nounwind {
entry:
	%dst_i.addr = alloca i64*		; <i64**> [#uses=2]
	%src_i.addr = alloca i64*		; <i64**> [#uses=2]
	%n.addr = alloca i32		; <i32*> [#uses=2]
	%i = alloca i32, align 4		; <i32*> [#uses=8]
	%dst = alloca <8 x i8>*, align 4		; <<8 x i8>**> [#uses=2]
	%src = alloca <8 x i8>*, align 4		; <<8 x i8>**> [#uses=2]
	store i64* %dst_i, i64** %dst_i.addr
	store i64* %src_i, i64** %src_i.addr
	store i32 %n, i32* %n.addr
	store i32 0, i32* %i
	br label %forcond

forcond:		; preds = %forinc, %entry
	%tmp = load i32* %i		; <i32> [#uses=1]
	%tmp1 = load i32* %n.addr		; <i32> [#uses=1]
	%cmp = icmp slt i32 %tmp, %tmp1		; <i1> [#uses=1]
	br i1 %cmp, label %forbody, label %afterfor

forbody:		; preds = %forcond
	%tmp2 = load i32* %i		; <i32> [#uses=1]
	%tmp3 = load i64** %dst_i.addr		; <i64*> [#uses=1]
	%arrayidx = getelementptr i64* %tmp3, i32 %tmp2		; <i64*> [#uses=1]
	%conv = bitcast i64* %arrayidx to <8 x i8>*		; <<8 x i8>*> [#uses=1]
	store <8 x i8>* %conv, <8 x i8>** %dst
	%tmp4 = load i32* %i		; <i32> [#uses=1]
	%tmp5 = load i64** %src_i.addr		; <i64*> [#uses=1]
	%arrayidx6 = getelementptr i64* %tmp5, i32 %tmp4		; <i64*> [#uses=1]
	%conv7 = bitcast i64* %arrayidx6 to <8 x i8>*		; <<8 x i8>*> [#uses=1]
	store <8 x i8>* %conv7, <8 x i8>** %src
	%tmp8 = load i32* %i		; <i32> [#uses=1]
	%tmp9 = load <8 x i8>** %dst		; <<8 x i8>*> [#uses=1]
	%arrayidx10 = getelementptr <8 x i8>* %tmp9, i32 %tmp8		; <<8 x i8>*> [#uses=1]
	%tmp11 = load i32* %i		; <i32> [#uses=1]
	%tmp12 = load <8 x i8>** %src		; <<8 x i8>*> [#uses=1]
	%arrayidx13 = getelementptr <8 x i8>* %tmp12, i32 %tmp11		; <<8 x i8>*> [#uses=1]
	%tmp14 = load <8 x i8>* %arrayidx13		; <<8 x i8>> [#uses=1]
	%add = add <8 x i8> %tmp14, < i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1 >		; <<8 x i8>> [#uses=1]
	%shr = ashr <8 x i8> %add, < i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2 >		; <<8 x i8>> [#uses=1]
	store <8 x i8> %shr, <8 x i8>* %arrayidx10
	br label %forinc

forinc:		; preds = %forbody
	%tmp15 = load i32* %i		; <i32> [#uses=1]
	%inc = add i32 %tmp15, 1		; <i32> [#uses=1]
	store i32 %inc, i32* %i
	br label %forcond

afterfor:		; preds = %forcond
	ret void
}

