; RUN: llvm-as < %s | opt -instcombine | llvm-dis | \
; RUN:    grep {ret i32 %.toremerge} | count 2
;; Simple sinking tests

; "if then else"
define i32 @test1(i1 %C) {
	%A = alloca i32
        br i1 %C, label %Cond, label %Cond2

Cond:
        store i32 -987654321, i32* %A
        br label %Cont

Cond2:
	store i32 47, i32* %A
	br label %Cont

Cont:
	%V = load i32* %A
	ret i32 %V
}

; "if then"
define i32 @test2(i1 %C) {
	%A = alloca i32
	store i32 47, i32* %A
        br i1 %C, label %Cond, label %Cont

Cond:
        store i32 -987654321, i32* %A
        br label %Cont

Cont:
	%V = load i32* %A
	ret i32 %V
}

