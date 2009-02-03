; Test folding of constantexpr geps into normal geps.
; RUN: llvm-as < %s | opt -instcombine | llvm-dis | grep {icmp eq i64 %X, -1}
; PR2235

%S = type { i32, [ 100 x i32] }

define i1 @test(i64 %X, %S* %P) {
        %A = getelementptr %S* %P, i32 0, i32 1, i64 %X
        %B = getelementptr %S* %P, i32 0, i32 0
	%C = icmp eq i32* %A, %B
	ret i1 %C
}

