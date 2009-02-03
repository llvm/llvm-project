; RUN: llvm-as < %s | opt -gvnpre | llvm-dis

define void @set_depth_values(i32 %level) {
cond_true90:		; preds = %cond_next84
	br i1 false, label %cond_true105, label %cond_true151

cond_true105:		; preds = %cond_true90
	%tmp132 = add i32 %level, -3		; <i32> [#uses=2]
	%tmp133 = icmp sgt i32 %tmp132, 0		; <i1> [#uses=1]
	%max134 = select i1 %tmp133, i32 %tmp132, i32 1		; <i32> [#uses=0]
	br label %cond_true151

cond_true151:		; preds = %cond_true140, %cond_true105
	%tmp153 = add i32 %level, -3		; <i32> [#uses=2]
	%tmp154 = icmp sgt i32 %tmp153, 0		; <i1> [#uses=1]
	%max155 = select i1 %tmp154, i32 %tmp153, i32 1		; <i32> [#uses=0]
	ret void
}
