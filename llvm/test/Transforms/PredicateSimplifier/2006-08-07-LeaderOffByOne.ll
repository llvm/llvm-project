; RUN: llvm-as < %s | opt -predsimplify -disable-output

define void @safe_strcpy(i32 %size1) {
entry:
	%tmp = icmp eq i32 %size1, 0		; <i1> [#uses=1]
	br i1 %tmp, label %return, label %strlen.exit
strlen.exit:		; preds = %entry
	%tmp.upgrd.1 = trunc i64 0 to i32		; <i32> [#uses=2]
	%tmp6 = icmp ult i32 %tmp.upgrd.1, %size1		; <i1> [#uses=1]
	br i1 %tmp6, label %cond_true7, label %cond_false19
cond_true7:		; preds = %strlen.exit
	%tmp9 = icmp eq i32 %tmp.upgrd.1, 0		; <i1> [#uses=1]
	br i1 %tmp9, label %cond_next15, label %cond_true10
cond_true10:		; preds = %cond_true7
	ret void
cond_next15:		; preds = %cond_true7
	ret void
cond_false19:		; preds = %strlen.exit
	ret void
return:		; preds = %entry
	ret void
}

