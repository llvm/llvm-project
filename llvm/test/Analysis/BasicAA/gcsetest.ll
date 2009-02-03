; Test that GCSE uses basicaa to do alias analysis, which is capable of 
; disambiguating some obvious cases.  All loads should be removable in 
; this testcase.

; RUN: llvm-as < %s | opt -basicaa -gvn -instcombine -dce \
; RUN: | llvm-dis | not grep load

@A = global i32 7
@B = global i32 8

define i32 @test() {
	%A1 = load i32* @A

	store i32 123, i32* @B  ; Store cannot alias @A

	%A2 = load i32* @A
	%X = sub i32 %A1, %A2
	ret i32 %X
}

define i32 @test2() {
        %A1 = load i32* @A
        br label %Loop
Loop:
        %AP = phi i32 [0, %0], [%X, %Loop]
        store i32 %AP, i32* @B  ; Store cannot alias @A

        %A2 = load i32* @A
        %X = sub i32 %A1, %A2
        %c = icmp eq i32 %X, 0
        br i1 %c, label %out, label %Loop

out:
        ret i32 %X
}

declare void @external()

define i32 @test3() {
	%X = alloca i32
	store i32 7, i32* %X
	call void @external()
	%V = load i32* %X
	ret i32 %V
}

