; RUN: llvm-as < %s | opt -predsimplify -disable-output

define void @readMotionInfoFromNAL() {
entry:
	br i1 false, label %bb2425, label %cond_next30
cond_next30:		; preds = %entry
	ret void
bb2418:		; preds = %bb2425
	ret void
bb2425:		; preds = %entry
	%tmp2427 = icmp sgt i32 0, 3		; <i1> [#uses=1]
	br i1 %tmp2427, label %cond_next2429, label %bb2418
cond_next2429:		; preds = %bb2425
	ret void
}

