; RUN: opt < %s -jump-threading -mem2reg -instcombine -simplifycfg -simplifycfg-require-and-preserve-domtree=1  -S | FileCheck %s

declare i32 @f1()
declare i32 @f2()
declare void @f3()

define i32 @test(i1 %cond, i1 %cond2) {
; CHECK: test
	br i1 %cond, label %T1, label %F1

; CHECK-NOT: T1
T1:
	%v1 = call i32 @f1()
	br label %Merge

F1:
	%v2 = call i32 @f2()
	br label %Merge

Merge:
; CHECK: Merge:
; CHECK: %v1 = call i32 @f1()
; CHECK-NEXT: br i1 %cond2
	%A = phi i1 [true, %T1], [false, %F1]
	%B = phi i32 [%v1, %T1], [%v2, %F1]
	%C = and i1 %A, %cond2
	br i1 %C, label %T2, label %F2

T2:
	call void @f3()
	ret i32 %B

F2:
	ret i32 %B
}
