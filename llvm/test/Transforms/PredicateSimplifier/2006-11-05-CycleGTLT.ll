; RUN: llvm-as < %s | opt -predsimplify -disable-output

define void @diff(i32 %N) {
entry:
	%tmp = icmp sgt i32 %N, 0		; <i1> [#uses=1]
	br i1 %tmp, label %bb519, label %bb744
bb519:		; preds = %entry
	%tmp720101 = icmp slt i32 %N, 0		; <i1> [#uses=1]
	br i1 %tmp720101, label %bb744, label %bb744
bb744:		; preds = %bb519, %bb519, %entry
	ret void
}

