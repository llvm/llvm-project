; RUN: llvm-as < %s | opt -licm | llvm-dis | %prcontext add 1 | grep preheader.loopexit: 

define void @test() {
loopentry.2.i:
	br i1 false, label %no_exit.1.i.preheader, label %loopentry.3.i.preheader
no_exit.1.i.preheader:		; preds = %loopentry.2.i
	br label %no_exit.1.i
no_exit.1.i:		; preds = %endif.8.i, %no_exit.1.i.preheader
	br i1 false, label %return.i, label %endif.8.i
endif.8.i:		; preds = %no_exit.1.i
	%inc.1.i = add i32 0, 1		; <i32> [#uses=1]
	br i1 false, label %no_exit.1.i, label %loopentry.3.i.preheader.loopexit
loopentry.3.i.preheader.loopexit:		; preds = %endif.8.i
	br label %loopentry.3.i.preheader
loopentry.3.i.preheader:		; preds = %loopentry.3.i.preheader.loopexit, %loopentry.2.i
	%arg_num.0.i.ph13000 = phi i32 [ 0, %loopentry.2.i ], [ %inc.1.i, %loopentry.3.i.preheader.loopexit ]		; <i32> [#uses=0]
	ret void
return.i:		; preds = %no_exit.1.i
	ret void
}

