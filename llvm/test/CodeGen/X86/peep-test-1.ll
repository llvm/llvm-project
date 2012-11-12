; RUN: llc < %s -march=x86 > %t
; RUN: grep dec %t | count 1
; RUN: not grep test %t
; RUN: not grep cmp %t

define void @foo(i32 %n, double* nocapture %p) nounwind {
	br label %bb

bb:
	%indvar = phi i32 [ 0, %0 ], [ %indvar.next, %bb ]
	%i.03 = sub i32 %n, %indvar
	%1 = getelementptr double* %p, i32 %i.03
	%2 = load double* %1, align 4
	%3 = fmul double %2, 2.930000e+00
	store double %3, double* %1, align 4
	%4 = add i32 %i.03, -1
	%phitmp = icmp slt i32 %4, 0
	%indvar.next = add i32 %indvar, 1
	br i1 %phitmp, label %bb, label %return

return:
	ret void
}
