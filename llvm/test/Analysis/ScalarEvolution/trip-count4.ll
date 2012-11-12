; RUN: opt < %s -analyze -scalar-evolution \
; RUN:   | grep "sext.*trunc.*Exits: 11"

; ScalarEvolution should be able to compute a loop exit value for %indvar.i8.

define void @another_count_down_signed(double* %d, i64 %n) nounwind {
entry:
	br label %loop

loop:		; preds = %loop, %entry
	%indvar = phi i64 [ %n, %entry ], [ %indvar.next, %loop ]		; <i64> [#uses=4]
	%s0 = shl i64 %indvar, 8		; <i64> [#uses=1]
	%indvar.i8 = ashr i64 %s0, 8		; <i64> [#uses=1]
	%t0 = getelementptr double* %d, i64 %indvar.i8		; <double*> [#uses=2]
	%t1 = load double* %t0		; <double> [#uses=1]
	%t2 = fmul double %t1, 1.000000e-01		; <double> [#uses=1]
	store double %t2, double* %t0
	%indvar.next = sub i64 %indvar, 1		; <i64> [#uses=2]
	%exitcond = icmp eq i64 %indvar.next, 10		; <i1> [#uses=1]
	br i1 %exitcond, label %return, label %loop

return:		; preds = %loop
	ret void
}
