; RUN: llvm-as < %s | opt -jump-threading | llvm-dis | not grep {br }
; RUN: llvm-as < %s | opt -jump-threading | llvm-dis | grep {ret i32} | count 1

define i32 @test(i1 %cond) {
	br i1 undef, label %T1, label %F1

T1:
	ret i32 42

F1:
	ret i32 17
}
