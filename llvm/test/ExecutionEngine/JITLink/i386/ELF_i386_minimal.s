# RUN: llvm-mc -triple=i386-unknown-linux-gnu -position-independent -filetype=obj -o %t.o %s
# RUN: llvm-jitlink -noexec %t.o

	.text
	.globl	main
	.p2align	4
	.type	main,@function
main:
    pushl   %ebp
    movl    %esp, %ebp
    pushl   %eax
    movl    $0, -4(%ebp)
    movl    $42, %eax
    addl    $4, %esp
    popl    %ebp
    retl

	.size	main, .-main