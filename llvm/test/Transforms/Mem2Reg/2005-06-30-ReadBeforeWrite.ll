; RUN: opt < %s -mem2reg -instcombine -S | grep store
; PR590


define void @zero(i8* %p, i32 %n) {
entry:
	%p_addr = alloca i8*		; <i8**> [#uses=2]
	%n_addr = alloca i32		; <i32*> [#uses=2]
	%i = alloca i32		; <i32*> [#uses=6]
	%out = alloca i32		; <i32*> [#uses=2]
	%undef = alloca i32		; <i32*> [#uses=2]
	store i8* %p, i8** %p_addr
	store i32 %n, i32* %n_addr
	store i32 0, i32* %i
	br label %loopentry
loopentry:		; preds = %endif, %entry
	%tmp.0 = load i32* %n_addr		; <i32> [#uses=1]
	%tmp.1 = add i32 %tmp.0, 1		; <i32> [#uses=1]
	%tmp.2 = load i32* %i		; <i32> [#uses=1]
	%tmp.3 = icmp sgt i32 %tmp.1, %tmp.2		; <i1> [#uses=2]
	%tmp.4 = zext i1 %tmp.3 to i32		; <i32> [#uses=0]
	br i1 %tmp.3, label %no_exit, label %return
no_exit:		; preds = %loopentry
	%tmp.5 = load i32* %undef		; <i32> [#uses=1]
	store i32 %tmp.5, i32* %out
	store i32 0, i32* %undef
	%tmp.6 = load i32* %i		; <i32> [#uses=1]
	%tmp.7 = icmp sgt i32 %tmp.6, 0		; <i1> [#uses=2]
	%tmp.8 = zext i1 %tmp.7 to i32		; <i32> [#uses=0]
	br i1 %tmp.7, label %then, label %endif
then:		; preds = %no_exit
	%tmp.9 = load i8** %p_addr		; <i8*> [#uses=1]
	%tmp.10 = load i32* %i		; <i32> [#uses=1]
	%tmp.11 = sub i32 %tmp.10, 1		; <i32> [#uses=1]
	%tmp.12 = getelementptr i8* %tmp.9, i32 %tmp.11		; <i8*> [#uses=1]
	%tmp.13 = load i32* %out		; <i32> [#uses=1]
	%tmp.14 = trunc i32 %tmp.13 to i8		; <i8> [#uses=1]
	store i8 %tmp.14, i8* %tmp.12
	br label %endif
endif:		; preds = %then, %no_exit
	%tmp.15 = load i32* %i		; <i32> [#uses=1]
	%inc = add i32 %tmp.15, 1		; <i32> [#uses=1]
	store i32 %inc, i32* %i
	br label %loopentry
return:		; preds = %loopentry
	ret void
}
