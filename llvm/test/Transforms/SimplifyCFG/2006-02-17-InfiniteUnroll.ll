; RUN: opt < %s -simplifycfg -disable-output

define void @polnel_() {
entry:
	%tmp595 = icmp slt i32 0, 0		; <i1> [#uses=4]
	br i1 %tmp595, label %bb148.critedge, label %cond_true40
bb36:		; preds = %bb43
	br i1 %tmp595, label %bb43, label %cond_true40
cond_true40:		; preds = %bb46, %cond_true40, %bb36, %entry
	%tmp397 = icmp sgt i32 0, 0		; <i1> [#uses=1]
	br i1 %tmp397, label %bb43, label %cond_true40
bb43:		; preds = %cond_true40, %bb36
	br i1 false, label %bb53, label %bb36
bb46:		; preds = %bb53
	br i1 %tmp595, label %bb53, label %cond_true40
bb53:		; preds = %bb46, %bb43
	br i1 false, label %bb102, label %bb46
bb92.preheader:		; preds = %bb102
	ret void
bb102:		; preds = %bb53
	br i1 %tmp595, label %bb148, label %bb92.preheader
bb148.critedge:		; preds = %entry
	ret void
bb148:		; preds = %bb102
	ret void
}

