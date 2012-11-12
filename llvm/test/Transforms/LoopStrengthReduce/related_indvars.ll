; RUN: opt < %s -loop-reduce -S | grep phi | count 1

; This should only result in one PHI node!

; void foo(double *D, double *E, double F) {
;   while (D != E)
;     *D++ = F;
; }

define void @foo(double* %D, double* %E, double %F) nounwind {
entry:
	%tmp.24 = icmp eq double* %D, %E		; <i1> [#uses=1]
	br i1 %tmp.24, label %return, label %no_exit
no_exit:		; preds = %no_exit, %entry
	%indvar = phi i32 [ 0, %entry ], [ %indvar.next, %no_exit ]		; <i32> [#uses=2]
	%D_addr.0.0.rec = bitcast i32 %indvar to i32		; <i32> [#uses=2]
	%D_addr.0.0 = getelementptr double* %D, i32 %D_addr.0.0.rec		; <double*> [#uses=1]
	%inc.rec = add i32 %D_addr.0.0.rec, 1		; <i32> [#uses=1]
	%inc = getelementptr double* %D, i32 %inc.rec		; <double*> [#uses=1]
	store double %F, double* %D_addr.0.0
	%tmp.2 = icmp eq double* %inc, %E		; <i1> [#uses=1]
	%indvar.next = add i32 %indvar, 1		; <i32> [#uses=1]
	br i1 %tmp.2, label %return, label %no_exit
return:		; preds = %no_exit, %entry
	ret void
}

