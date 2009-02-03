; RUN: llvm-as < %s | llc -march=x86-64 | grep {leal	(%rdi,%rdi,2), %eax}
define i32 @test(i32 %a) {
        %tmp2 = mul i32 %a, 3           ; <i32> [#uses=1]
        ret i32 %tmp2
}

; RUN: llvm-as < %s | llc -march=x86-64 | grep {leaq	(,%rdi,4), %rax}
define i64 @test2(i64 %a) {
        %tmp2 = shl i64 %a, 2
	%tmp3 = or i64 %tmp2, %a
        ret i64 %tmp3
}

;; TODO!  LEA instead of shift + copy.
define i64 @test3(i64 %a) {
        %tmp2 = shl i64 %a, 3
        ret i64 %tmp2
}

