; There should be no phi nodes left.
; RUN: llvm-as < %s | opt -jump-threading -simplifycfg -mem2reg | llvm-dis | not grep {phi i32}

declare i32 @f1()
declare i32 @f2()
declare void @f3()

define i32 @test(i1 %cond) {
	br i1 %cond, label %T1, label %F1

T1:
	%v1 = call i32 @f1()
	br label %Merge

F1:
	%v2 = call i32 @f2()
	br label %Merge

Merge:
	%B = phi i32 [%v1, %T1], [12, %F1]
	%A = icmp ne i32 %B, 42
	br i1 %A, label %T2, label %F2

T2:
	call void @f3()
	ret i32 1

F2:
	ret i32 0
}
