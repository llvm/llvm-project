; RUN: llvm-as < %s | opt -tailduplicate -disable-output

define void @interpret() {
entry:
	br label %retry
retry:		; preds = %endif.4, %entry
	%tmp.8 = call i32 @interp( )		; <i32> [#uses=3]
	switch i32 0, label %endif.4 [
		 i32 -25, label %return
		 i32 -16, label %return
	]
endif.4:		; preds = %retry
	br i1 false, label %return, label %retry
return:		; preds = %endif.4, %retry, %retry
	%result.0 = phi i32 [ %tmp.8, %retry ], [ %tmp.8, %retry ], [ %tmp.8, %endif.4 ]		; <i32> [#uses=0]
	ret void
}

declare i32 @interp()

