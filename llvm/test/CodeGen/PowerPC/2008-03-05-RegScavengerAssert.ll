; RUN: llc < %s -mtriple=powerpc-apple-darwin

declare i8* @bar(i32)

define void @foo(i8* %pp) nounwind  {
entry:
	%tmp2 = tail call i8* @bar( i32 14 ) nounwind 		; <i8*> [#uses=0]
	%tmp28 = bitcast i8* %pp to void ()**		; <void ()**> [#uses=1]
	%tmp38 = load void ()** %tmp28, align 4		; <void ()*> [#uses=2]
	br i1 false, label %bb34, label %bb25
bb25:		; preds = %entry
	%tmp30 = bitcast void ()* %tmp38 to void (i8*)*		; <void (i8*)*> [#uses=1]
	tail call void %tmp30( i8* null ) nounwind 
	ret void
bb34:		; preds = %entry
	tail call void %tmp38( ) nounwind 
	ret void
}
