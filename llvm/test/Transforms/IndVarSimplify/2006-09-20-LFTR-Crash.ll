; RUN: opt < %s -indvars -disable-output
; ModuleID = '2006-09-20-LFTR-Crash.ll'
	%struct.p7prior_s = type { i32, i32, [200 x float], [200 x [7 x float]], i32, [200 x float], [200 x [20 x float]], i32, [200 x float], [200 x [20 x float]] }

define void @P7DefaultPrior() {
entry:
	switch i32 0, label %UnifiedReturnBlock [
		 i32 2, label %bb160
		 i32 3, label %bb
	]

bb:		; preds = %entry
	br i1 false, label %cond_true.i, label %sre_malloc.exit

cond_true.i:		; preds = %bb
	unreachable

sre_malloc.exit:		; preds = %bb
	br label %cond_true

cond_true:		; preds = %cond_true66, %cond_true, %sre_malloc.exit
	%tmp59 = phi i32 [ 1, %sre_malloc.exit ], [ %phitmp, %cond_true66 ], [ %tmp59, %cond_true ]		; <i32> [#uses=2]
	%indvar245.0.ph = phi i32 [ 0, %sre_malloc.exit ], [ %indvar.next246, %cond_true66 ], [ %indvar245.0.ph, %cond_true ]		; <i32> [#uses=2]
	br i1 false, label %bb57, label %cond_true

bb57:		; preds = %cond_true
	%tmp65 = icmp sgt i32 0, %tmp59		; <i1> [#uses=1]
	%indvar.next246 = add i32 %indvar245.0.ph, 1		; <i32> [#uses=2]
	br i1 %tmp65, label %cond_true66, label %bb69

cond_true66:		; preds = %bb57
	%q.1.0 = bitcast i32 %indvar.next246 to i32		; <i32> [#uses=1]
	%phitmp = add i32 %q.1.0, 1		; <i32> [#uses=1]
	br label %cond_true

bb69:		; preds = %bb57
	ret void

bb160:		; preds = %entry
	ret void

UnifiedReturnBlock:		; preds = %entry
	ret void
}
