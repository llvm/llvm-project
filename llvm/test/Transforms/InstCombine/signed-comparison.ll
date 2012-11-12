; RUN: opt < %s -instcombine -S > %t
; RUN: not grep zext %t
; RUN: not grep slt %t
; RUN: grep "icmp ult" %t

; Instcombine should convert the zext+slt into a simple ult.

define void @foo(double* %p) nounwind {
entry:
	br label %bb

bb:
	%indvar = phi i64 [ 0, %entry ], [ %indvar.next, %bb ]
	%t0 = and i64 %indvar, 65535
	%t1 = getelementptr double* %p, i64 %t0
	%t2 = load double* %t1, align 8
	%t3 = fmul double %t2, 2.2
	store double %t3, double* %t1, align 8
	%i.04 = trunc i64 %indvar to i16
	%t4 = add i16 %i.04, 1
	%t5 = zext i16 %t4 to i32
	%t6 = icmp slt i32 %t5, 500
	%indvar.next = add i64 %indvar, 1
	br i1 %t6, label %bb, label %return

return:
	ret void
}
