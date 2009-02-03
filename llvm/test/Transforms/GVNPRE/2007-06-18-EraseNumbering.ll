; RUN: llvm-as < %s | opt -gvnpre | llvm-dis

define i32 @TreeCCStreamFlush(i8* %stream) {
entry:
	br i1 false, label %bb55.preheader, label %cond_true

cond_true:		; preds = %entry
	ret i32 0

bb55.preheader:		; preds = %entry
	%tmp57 = icmp eq i8* null, null		; <i1> [#uses=0]
	br i1 false, label %cond_next106, label %bb124

cond_next106:		; preds = %bb55.preheader
	%tmp109 = load i8** null
	br i1 false, label %bb124, label %bb116

bb116:		; preds = %cond_next106
	ret i32 0

bb124:		; preds = %cond_next106, %bb55.preheader
	%buffer.4 = phi i8* [ null, %bb55.preheader ], [ %tmp109, %cond_next106 ]
	%tmp131 = icmp eq i8* %buffer.4, null
	%bothcond = or i1 %tmp131, false
	ret i32 0
}
