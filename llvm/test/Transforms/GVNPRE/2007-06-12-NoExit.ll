; RUN: llvm-as < %s | opt -gvnpre | llvm-dis

define void @_Z4sortI3Lit16LessThan_defaultIS0_EEvPT_iT0_() {
entry:
	br label %cond_false

cond_false:		; preds = %cond_false, %entry
	br label %cond_false
}
