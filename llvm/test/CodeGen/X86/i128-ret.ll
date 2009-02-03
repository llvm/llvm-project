; RUN: llvm-as < %s | llc -march=x86-64 | grep {movq	8(%rdi), %rdx}
; RUN: llvm-as < %s | llc -march=x86-64 | grep {movq	(%rdi), %rax}

define i128 @test(i128 *%P) {
        %A = load i128* %P
        ret i128 %A
}

