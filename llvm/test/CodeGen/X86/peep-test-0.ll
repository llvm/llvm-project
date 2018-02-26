; RUN: llc < %s -mtriple=x86_64-- > %t
; RUN: not grep cmp %t
; RUN: not grep test %t

define void @loop(i64 %n, double* nocapture %d) nounwind {
entry:
	br label %bb

bb:
	%indvar = phi i64 [ %n, %entry ], [ %indvar.next, %bb ]
	%i.03 = add i64 %indvar, %n
	%0 = getelementptr double, double* %d, i64 %i.03
	%1 = load double, double* %0, align 8
	%2 = fmul double %1, 3.000000e+00
	store double %2, double* %0, align 8
	%indvar.next = add i64 %indvar, 1
	%exitcond = icmp eq i64 %indvar.next, 0
	br i1 %exitcond, label %return, label %bb

return:
	ret void
}
