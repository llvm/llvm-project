; RUN: llc < %s -mcpu=generic -march=x86-64 > %t
; RUN: not grep and %t
; RUN: not grep movz %t
; RUN: not grep sar %t
; RUN: not grep shl %t
; RUN: grep add %t | count 1
; RUN: grep inc %t | count 4
; RUN: grep dec %t | count 2
; RUN: grep lea %t | count 3

; Optimize away zext-inreg and sext-inreg on the loop induction
; variable using trip-count information.

define void @count_up(double* %d, i64 %n) nounwind {
entry:
	br label %loop

loop:
	%indvar = phi i64 [ 0, %entry ], [ %indvar.next, %loop ]
	%indvar.i8 = and i64 %indvar, 255
	%t0 = getelementptr double* %d, i64 %indvar.i8
	%t1 = load double* %t0
	%t2 = fmul double %t1, 0.1
	store double %t2, double* %t0
	%indvar.i24 = and i64 %indvar, 16777215
	%t3 = getelementptr double* %d, i64 %indvar.i24
	%t4 = load double* %t3
	%t5 = fmul double %t4, 2.3
	store double %t5, double* %t3
	%t6 = getelementptr double* %d, i64 %indvar
	%t7 = load double* %t6
	%t8 = fmul double %t7, 4.5
	store double %t8, double* %t6
	%indvar.next = add i64 %indvar, 1
	%exitcond = icmp eq i64 %indvar.next, 10
	br i1 %exitcond, label %return, label %loop

return:
	ret void
}

define void @count_down(double* %d, i64 %n) nounwind {
entry:
	br label %loop

loop:
	%indvar = phi i64 [ 10, %entry ], [ %indvar.next, %loop ]
	%indvar.i8 = and i64 %indvar, 255
	%t0 = getelementptr double* %d, i64 %indvar.i8
	%t1 = load double* %t0
	%t2 = fmul double %t1, 0.1
	store double %t2, double* %t0
	%indvar.i24 = and i64 %indvar, 16777215
	%t3 = getelementptr double* %d, i64 %indvar.i24
	%t4 = load double* %t3
	%t5 = fmul double %t4, 2.3
	store double %t5, double* %t3
	%t6 = getelementptr double* %d, i64 %indvar
	%t7 = load double* %t6
	%t8 = fmul double %t7, 4.5
	store double %t8, double* %t6
	%indvar.next = sub i64 %indvar, 1
	%exitcond = icmp eq i64 %indvar.next, 0
	br i1 %exitcond, label %return, label %loop

return:
	ret void
}

define void @count_up_signed(double* %d, i64 %n) nounwind {
entry:
	br label %loop

loop:
	%indvar = phi i64 [ 0, %entry ], [ %indvar.next, %loop ]
        %s0 = shl i64 %indvar, 8
	%indvar.i8 = ashr i64 %s0, 8
	%t0 = getelementptr double* %d, i64 %indvar.i8
	%t1 = load double* %t0
	%t2 = fmul double %t1, 0.1
	store double %t2, double* %t0
	%s1 = shl i64 %indvar, 24
	%indvar.i24 = ashr i64 %s1, 24
	%t3 = getelementptr double* %d, i64 %indvar.i24
	%t4 = load double* %t3
	%t5 = fmul double %t4, 2.3
	store double %t5, double* %t3
	%t6 = getelementptr double* %d, i64 %indvar
	%t7 = load double* %t6
	%t8 = fmul double %t7, 4.5
	store double %t8, double* %t6
	%indvar.next = add i64 %indvar, 1
	%exitcond = icmp eq i64 %indvar.next, 10
	br i1 %exitcond, label %return, label %loop

return:
	ret void
}

define void @count_down_signed(double* %d, i64 %n) nounwind {
entry:
	br label %loop

loop:
	%indvar = phi i64 [ 10, %entry ], [ %indvar.next, %loop ]
        %s0 = shl i64 %indvar, 8
	%indvar.i8 = ashr i64 %s0, 8
	%t0 = getelementptr double* %d, i64 %indvar.i8
	%t1 = load double* %t0
	%t2 = fmul double %t1, 0.1
	store double %t2, double* %t0
	%s1 = shl i64 %indvar, 24
	%indvar.i24 = ashr i64 %s1, 24
	%t3 = getelementptr double* %d, i64 %indvar.i24
	%t4 = load double* %t3
	%t5 = fmul double %t4, 2.3
	store double %t5, double* %t3
	%t6 = getelementptr double* %d, i64 %indvar
	%t7 = load double* %t6
	%t8 = fmul double %t7, 4.5
	store double %t8, double* %t6
	%indvar.next = sub i64 %indvar, 1
	%exitcond = icmp eq i64 %indvar.next, 0
	br i1 %exitcond, label %return, label %loop

return:
	ret void
}

define void @another_count_up(double* %d, i64 %n) nounwind {
entry:
	br label %loop

loop:
	%indvar = phi i64 [ 18446744073709551615, %entry ], [ %indvar.next, %loop ]
	%indvar.i8 = and i64 %indvar, 255
	%t0 = getelementptr double* %d, i64 %indvar.i8
	%t1 = load double* %t0
	%t2 = fmul double %t1, 0.1
	store double %t2, double* %t0
	%indvar.i24 = and i64 %indvar, 16777215
	%t3 = getelementptr double* %d, i64 %indvar.i24
	%t4 = load double* %t3
	%t5 = fmul double %t4, 2.3
	store double %t5, double* %t3
	%t6 = getelementptr double* %d, i64 %indvar
	%t7 = load double* %t6
	%t8 = fmul double %t7, 4.5
	store double %t8, double* %t6
	%indvar.next = add i64 %indvar, 1
	%exitcond = icmp eq i64 %indvar.next, 0
	br i1 %exitcond, label %return, label %loop

return:
	ret void
}

define void @another_count_down(double* %d, i64 %n) nounwind {
entry:
	br label %loop

loop:
	%indvar = phi i64 [ 0, %entry ], [ %indvar.next, %loop ]
	%indvar.i8 = and i64 %indvar, 255
	%t0 = getelementptr double* %d, i64 %indvar.i8
	%t1 = load double* %t0
	%t2 = fmul double %t1, 0.1
	store double %t2, double* %t0
	%indvar.i24 = and i64 %indvar, 16777215
	%t3 = getelementptr double* %d, i64 %indvar.i24
	%t4 = load double* %t3
	%t5 = fdiv double %t4, 2.3
	store double %t5, double* %t3
	%t6 = getelementptr double* %d, i64 %indvar
	%t7 = load double* %t6
	%t8 = fmul double %t7, 4.5
	store double %t8, double* %t6
	%indvar.next = sub i64 %indvar, 1
	%exitcond = icmp eq i64 %indvar.next, 18446744073709551615
	br i1 %exitcond, label %return, label %loop

return:
	ret void
}

define void @another_count_up_signed(double* %d, i64 %n) nounwind {
entry:
	br label %loop

loop:
	%indvar = phi i64 [ 18446744073709551615, %entry ], [ %indvar.next, %loop ]
        %s0 = shl i64 %indvar, 8
	%indvar.i8 = ashr i64 %s0, 8
	%t0 = getelementptr double* %d, i64 %indvar.i8
	%t1 = load double* %t0
	%t2 = fmul double %t1, 0.1
	store double %t2, double* %t0
	%s1 = shl i64 %indvar, 24
	%indvar.i24 = ashr i64 %s1, 24
	%t3 = getelementptr double* %d, i64 %indvar.i24
	%t4 = load double* %t3
	%t5 = fdiv double %t4, 2.3
	store double %t5, double* %t3
	%t6 = getelementptr double* %d, i64 %indvar
	%t7 = load double* %t6
	%t8 = fmul double %t7, 4.5
	store double %t8, double* %t6
	%indvar.next = add i64 %indvar, 1
	%exitcond = icmp eq i64 %indvar.next, 0
	br i1 %exitcond, label %return, label %loop

return:
	ret void
}

define void @another_count_down_signed(double* %d, i64 %n) nounwind {
entry:
	br label %loop

loop:
	%indvar = phi i64 [ 0, %entry ], [ %indvar.next, %loop ]
        %s0 = shl i64 %indvar, 8
	%indvar.i8 = ashr i64 %s0, 8
	%t0 = getelementptr double* %d, i64 %indvar.i8
	%t1 = load double* %t0
	%t2 = fmul double %t1, 0.1
	store double %t2, double* %t0
	%s1 = shl i64 %indvar, 24
	%indvar.i24 = ashr i64 %s1, 24
	%t3 = getelementptr double* %d, i64 %indvar.i24
	%t4 = load double* %t3
	%t5 = fdiv double %t4, 2.3
	store double %t5, double* %t3
	%t6 = getelementptr double* %d, i64 %indvar
	%t7 = load double* %t6
	%t8 = fmul double %t7, 4.5
	store double %t8, double* %t6
	%indvar.next = sub i64 %indvar, 1
	%exitcond = icmp eq i64 %indvar.next, 18446744073709551615
	br i1 %exitcond, label %return, label %loop

return:
	ret void
}
