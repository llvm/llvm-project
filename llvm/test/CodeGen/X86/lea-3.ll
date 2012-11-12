; RUN: llc < %s -mtriple=x86_64-linux | FileCheck %s
; RUN: llc < %s -mtriple=x86_64-win32 | FileCheck %s

; CHECK: leaq (,[[A0:%rdi|%rcx]],4), %rax
define i64 @test2(i64 %a) {
        %tmp2 = shl i64 %a, 2
	%tmp3 = or i64 %tmp2, %a
        ret i64 %tmp3
}

; CHECK: leal ([[A0]],[[A0]],2), %eax
define i32 @test(i32 %a) {
        %tmp2 = mul i32 %a, 3           ; <i32> [#uses=1]
        ret i32 %tmp2
}

; CHECK: leaq (,[[A0]],8), %rax
define i64 @test3(i64 %a) {
        %tmp2 = shl i64 %a, 3
        ret i64 %tmp2
}

