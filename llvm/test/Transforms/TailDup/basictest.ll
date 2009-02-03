; RUN: llvm-as < %s | opt -tailduplicate -disable-output

declare void @__main()

define i32 @main() {
entry:
	call void @__main( )
	br label %loopentry
loopentry:		; preds = %no_exit, %entry
	%i.0 = phi i32 [ %inc, %no_exit ], [ 0, %entry ]		; <i32> [#uses=3]
	%tmp.1 = icmp sle i32 %i.0, 99		; <i1> [#uses=1]
	br i1 %tmp.1, label %no_exit, label %return
no_exit:		; preds = %loopentry
	%tmp.51 = call i32 @main( )		; <i32> [#uses=0]
	%inc = add i32 %i.0, 1		; <i32> [#uses=1]
	br label %loopentry
return:		; preds = %loopentry
	ret i32 %i.0
}

