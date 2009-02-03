; This testcase makes sure that size is taken to account when alias analysis 
; is performed.  It is not legal to delete the second load instruction because
; the value computed by the first load instruction is changed by the store.

; RUN: llvm-as < %s | opt -gvn -instcombine | llvm-dis | grep DONOTREMOVE

define i32 @test() {
	%A = alloca i32
	store i32 0, i32* %A
    %X = load i32* %A
    %B = bitcast i32* %A to i8*
    %C = getelementptr i8* %B, i64 1
	store i8 1, i8* %C    ; Aliases %A
    %Y.DONOTREMOVE = load i32* %A
	%Z = sub i32 %X, %Y.DONOTREMOVE
    ret i32 %Z
}

