; RUN: llvm-as < %s | opt -postdomfrontier -disable-output

define void @args_out_of_range() {
entry:
	br label %bb

bb:		; preds = %bb, %entry
	br label %bb
}

define void @args_out_of_range_3() {
entry:
	br label %bb

bb:		; preds = %bb, %entry
	br label %bb
}

define void @Feq() {
entry:
	br i1 false, label %cond_true, label %cond_next

cond_true:		; preds = %entry
	unreachable

cond_next:		; preds = %entry
	unreachable
}
