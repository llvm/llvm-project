; RUN: llvm-as < %s | opt -tailduplicate -disable-output

define i32 @foo() {
entry:
	br label %return.i
after_ret.i:		; No predecessors!
	br label %return.i
return.i:		; preds = %after_ret.i, %entry
	%tmp.3 = ptrtoint i32* null to i32		; <i32> [#uses=1]
	br label %return.i1
after_ret.i1:		; No predecessors!
	br label %return.i1
return.i1:		; preds = %after_ret.i1, %return.i
	%tmp.8 = sub i32 %tmp.3, 0		; <i32> [#uses=0]
	ret i32 0
}

