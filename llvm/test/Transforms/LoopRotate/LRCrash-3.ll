; RUN: llvm-as < %s | opt -loop-rotate -disable-output

define void @_ZN9Classfile4readEv() {
entry:
	br i1 false, label %cond_false485, label %bb405
bb405:		; preds = %entry
	ret void
cond_false485:		; preds = %entry
	br label %bb830
bb511:		; preds = %bb830
	br i1 false, label %bb816, label %bb830
cond_next667:		; preds = %bb816
	br i1 false, label %cond_next695, label %bb680
bb676:		; preds = %bb680
	br label %bb680
bb680:		; preds = %bb676, %cond_next667
	%iftmp.68.0 = zext i1 false to i8		; <i8> [#uses=1]
	br i1 false, label %bb676, label %cond_next695
cond_next695:		; preds = %bb680, %cond_next667
	%iftmp.68.2 = phi i8 [ %iftmp.68.0, %bb680 ], [ undef, %cond_next667 ]		; <i8> [#uses=0]
	ret void
bb816:		; preds = %bb816, %bb511
	br i1 false, label %cond_next667, label %bb816
bb830:		; preds = %bb511, %cond_false485
	br i1 false, label %bb511, label %bb835
bb835:		; preds = %bb830
	ret void
}

