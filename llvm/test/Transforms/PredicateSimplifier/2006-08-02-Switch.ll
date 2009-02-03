; RUN: llvm-as < %s | opt -predsimplify -disable-output

define fastcc void @_ov_splice(i32 %n1, i32 %n2, i32 %ch2) {
entry:
	%tmp = icmp sgt i32 %n1, %n2		; <i1> [#uses=1]
	%n.0 = select i1 %tmp, i32 %n2, i32 %n1		; <i32> [#uses=1]
	%tmp104 = icmp slt i32 0, %ch2		; <i1> [#uses=1]
	br i1 %tmp104, label %cond_true105, label %return
cond_true95:		; preds = %cond_true105
	ret void
bb98:		; preds = %cond_true105
	ret void
cond_true105:		; preds = %entry
	%tmp94 = icmp sgt i32 %n.0, 0		; <i1> [#uses=1]
	br i1 %tmp94, label %cond_true95, label %bb98
return:		; preds = %entry
	ret void
}

