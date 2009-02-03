; RUN: llvm-as < %s | opt -predsimplify -disable-output

define void @gs_image_next() {
entry:
	%tmp = load i32* null		; <i32> [#uses=2]
	br i1 false, label %cond_next21, label %UnifiedReturnBlock
cond_next21:		; preds = %entry
	br i1 false, label %cond_next42, label %UnifiedReturnBlock
cond_next42:		; preds = %cond_next21
	br label %cond_true158
cond_next134:		; preds = %cond_true158
	%tmp1571 = icmp eq i32 0, %min		; <i1> [#uses=0]
	ret void
cond_true158:		; preds = %cond_true158, %cond_next42
	%tmp47 = sub i32 %tmp, 0		; <i32> [#uses=2]
	%tmp49 = icmp ule i32 %tmp47, 0		; <i1> [#uses=1]
	%min = select i1 %tmp49, i32 %tmp47, i32 0		; <i32> [#uses=2]
	%tmp92 = add i32 %min, 0		; <i32> [#uses=1]
	%tmp101 = icmp eq i32 %tmp92, %tmp		; <i1> [#uses=1]
	br i1 %tmp101, label %cond_next134, label %cond_true158
UnifiedReturnBlock:		; preds = %cond_next21, %entry
	ret void
}

