; RUN: llvm-as < %s | llc -march=x86 | not grep lea
; RUN: llvm-as < %s | llc -march=x86-64 | not grep lea

@B = external global [1000 x i8], align 32
@A = external global [1000 x i8], align 32
@P = external global [1000 x i8], align 32

define void @foo(i32 %m) {
entry:
	%tmp1 = icmp sgt i32 %m, 0
	br i1 %tmp1, label %bb, label %return

bb:
	%i.019.0 = phi i32 [ %indvar.next, %bb ], [ 0, %entry ]
	%tmp2 = getelementptr [1000 x i8]* @B, i32 0, i32 %i.019.0
	%tmp3 = load i8* %tmp2, align 4
	%tmp4 = mul i8 %tmp3, 2
	%tmp5 = getelementptr [1000 x i8]* @A, i32 0, i32 %i.019.0
	store i8 %tmp4, i8* %tmp5, align 4
	%tmp8 = mul i32 %i.019.0, 9
	%tmp10 = getelementptr [1000 x i8]* @P, i32 0, i32 %tmp8
	store i8 17, i8* %tmp10, align 4
	%indvar.next = add i32 %i.019.0, 1
	%exitcond = icmp eq i32 %indvar.next, %m
	br i1 %exitcond, label %return, label %bb

return:
	ret void
}

