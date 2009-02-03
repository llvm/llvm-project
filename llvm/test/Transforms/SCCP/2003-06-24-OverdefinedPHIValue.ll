; RUN: llvm-as < %s | opt -sccp -simplifycfg | llvm-dis | \
; RUN:   not grep then:

define void @cprop_test11(i32* %data.1) {
entry:
	%tmp.1 = load i32* %data.1		; <i32> [#uses=3]
	%tmp.41 = icmp sgt i32 %tmp.1, 1		; <i1> [#uses=1]
	br i1 %tmp.41, label %no_exit, label %loopexit
no_exit:		; preds = %endif, %then, %entry
	%j.0 = phi i32 [ %j.0, %endif ], [ %i.0, %then ], [ 1, %entry ]		; <i32> [#uses=3]
	%i.0 = phi i32 [ %inc, %endif ], [ %inc1, %then ], [ 1, %entry ]		; <i32> [#uses=4]
	%tmp.8.not = icmp ne i32 %j.0, 0		; <i1> [#uses=1]
	br i1 %tmp.8.not, label %endif, label %then
then:		; preds = %no_exit
	%inc1 = add i32 %i.0, 1		; <i32> [#uses=3]
	%tmp.42 = icmp slt i32 %inc1, %tmp.1		; <i1> [#uses=1]
	br i1 %tmp.42, label %no_exit, label %loopexit
endif:		; preds = %no_exit
	%inc = add i32 %i.0, 1		; <i32> [#uses=3]
	%tmp.4 = icmp slt i32 %inc, %tmp.1		; <i1> [#uses=1]
	br i1 %tmp.4, label %no_exit, label %loopexit
loopexit:		; preds = %endif, %then, %entry
	%j.1 = phi i32 [ 1, %entry ], [ %j.0, %endif ], [ %i.0, %then ]		; <i32> [#uses=1]
	%i.1 = phi i32 [ 1, %entry ], [ %inc, %endif ], [ %inc1, %then ]		; <i32> [#uses=1]
	%tmp.17 = getelementptr i32* %data.1, i64 1		; <i32*> [#uses=1]
	store i32 %j.1, i32* %tmp.17
	%tmp.23 = getelementptr i32* %data.1, i64 2		; <i32*> [#uses=1]
	store i32 %i.1, i32* %tmp.23
	ret void
}
