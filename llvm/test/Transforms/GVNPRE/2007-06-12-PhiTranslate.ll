; RUN: llvm-as < %s | opt -gvnpre | llvm-dis

define void @strength_test5(i32* %data) {
entry:
	br i1 false, label %cond_next16.preheader, label %cond_true

cond_true:		; preds = %entry
	%tmp12 = icmp sgt i32 0, 0		; <i1> [#uses=1]
	br i1 %tmp12, label %return, label %cond_next16.preheader

cond_next16.preheader:		; preds = %cond_true, %entry
	%i.01.1.ph = phi i32 [ 1, %entry ], [ 1, %cond_true ]		; <i32> [#uses=1]
	%i.01.1 = add i32 0, %i.01.1.ph		; <i32> [#uses=0]
	%indvar.next = add i32 0, 1		; <i32> [#uses=0]
	ret void

return:		; preds = %cond_true
	ret void
}
