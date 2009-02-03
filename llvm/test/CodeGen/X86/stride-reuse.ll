; RUN: llvm-as < %s | llc -march=x86 | not grep lea
; RUN: llvm-as < %s | llc -march=x86-64 | not grep lea

@B = external global [1000 x float], align 32
@A = external global [1000 x float], align 32
@P = external global [1000 x i32], align 32

define void @foo(i32 %m) {
entry:
	%tmp1 = icmp sgt i32 %m, 0
	br i1 %tmp1, label %bb, label %return

bb:
	%i.019.0 = phi i32 [ %indvar.next, %bb ], [ 0, %entry ]
	%tmp2 = getelementptr [1000 x float]* @B, i32 0, i32 %i.019.0
	%tmp3 = load float* %tmp2, align 4
	%tmp4 = mul float %tmp3, 2.000000e+00
	%tmp5 = getelementptr [1000 x float]* @A, i32 0, i32 %i.019.0
	store float %tmp4, float* %tmp5, align 4
	%tmp8 = shl i32 %i.019.0, 1
	%tmp9 = add i32 %tmp8, 64
	%tmp10 = getelementptr [1000 x i32]* @P, i32 0, i32 %i.019.0
	store i32 %tmp9, i32* %tmp10, align 4
	%indvar.next = add i32 %i.019.0, 1
	%exitcond = icmp eq i32 %indvar.next, %m
	br i1 %exitcond, label %return, label %bb

return:
	ret void
}
