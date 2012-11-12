; RUN: not llvm-as < %s 2>&1 | grep "multiple entries for the same basic block"



define i32 @test(i32 %i, i32 %j, i1 %c) {
	br i1 %c, label %A, label %A
A:
	%a = phi i32 [%i, %0], [%j, %0]  ; Error, different values from same block!
	ret i32 %a
}
