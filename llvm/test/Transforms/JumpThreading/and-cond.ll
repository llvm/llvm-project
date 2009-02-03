; RUN: llvm-as < %s | opt -jump-threading -mem2reg -instcombine -simplifycfg  | llvm-dis | grep {ret i32 %v1}
; There should be no uncond branches left.
; RUN: llvm-as < %s | opt -jump-threading -mem2reg -instcombine -simplifycfg  | llvm-dis | not grep {br label}

declare i32 @f1()
declare i32 @f2()
declare void @f3()

define i32 @test(i1 %cond, i1 %cond2) {
	br i1 %cond, label %T1, label %F1

T1:
	%v1 = call i32 @f1()
	br label %Merge

F1:
	%v2 = call i32 @f2()
	br label %Merge

Merge:
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
