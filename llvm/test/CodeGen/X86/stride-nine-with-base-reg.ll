; RUN: llc < %s -mtriple=i686-- -relocation-model=static | FileCheck %s
; RUN: llc < %s -mtriple=x86_64-linux               | FileCheck %s
; CHECK-NOT:     lea

; P should be sunk into the loop and folded into the address mode. There
; shouldn't be any lea instructions inside the loop.

@B = external dso_local global [1000 x i8], align 32
@A = external dso_local global [1000 x i8], align 32
@P = external dso_local global [1000 x i8], align 32
@Q = external dso_local global [1000 x i8], align 32

define void @foo(i32 %m, i32 %p) nounwind {
entry:
	%tmp1 = icmp sgt i32 %m, 0
	br i1 %tmp1, label %bb, label %return

bb:
	%i.019.0 = phi i32 [ %indvar.next, %bb ], [ 0, %entry ]
	%tmp2 = getelementptr [1000 x i8], ptr @B, i32 0, i32 %i.019.0
	%tmp3 = load i8, ptr %tmp2, align 4
	%tmp4 = mul i8 %tmp3, 2
	%tmp5 = getelementptr [1000 x i8], ptr @A, i32 0, i32 %i.019.0
	store i8 %tmp4, ptr %tmp5, align 4
	%tmp8 = mul i32 %i.019.0, 9
        %tmp0 = add i32 %tmp8, %p
	%tmp10 = getelementptr [1000 x i8], ptr @P, i32 0, i32 %tmp0
	store i8 17, ptr %tmp10, align 4
	%tmp11 = getelementptr [1000 x i8], ptr @Q, i32 0, i32 %tmp0
	store i8 19, ptr %tmp11, align 4
	%indvar.next = add i32 %i.019.0, 1
	%exitcond = icmp eq i32 %indvar.next, %m
	br i1 %exitcond, label %return, label %bb

return:
	ret void
}

