# REQUIRES: native && x86_64-linux && intel-jitevents

# RUN: rm -rf %t && mkdir %t
# RUN: llvm-mc -triple=x86_64-unknown-linux \
# RUN:     -filetype=obj -o %t/ELF_x86-64_vtune.o %s
# RUN: llvm-jitlink -vtune-support %t/ELF_x86-64_vtune.o | \
# RUN: FileCheck %s

# CHECK: Method load [0]: {{.*}}, Size = {{[0-9]+}}
# CHECK: Method unload [0]
        .file   "test.c"
        .text
        .globl  main
        .type   main, @function
main:
.LFB0:
        .cfi_startproc
        endbr64
        pushq   %rbp
        .cfi_def_cfa_offset 16
        .cfi_offset 6, -16
        movq    %rsp, %rbp
        .cfi_def_cfa_register 6
        movl    %edi, -4(%rbp)
        movq    %rsi, -16(%rbp)
        movl    -4(%rbp), %ebx
        addl    $1, %ebx
	movl   $0, %eax
        popq    %rbp
        .cfi_def_cfa 7, 8
        ret
        .cfi_endproc
.LFE0:
        .size   main, .-main
        .ident  "GCC: (Ubuntu 9.4.0-1ubuntu1~20.04.2) 9.4.0"
        .section        .note.GNU-stack,"",@progbits
        .section        .note.gnu.property,"a"
        .align 8
        .long    1f - 0f
        .long    4f - 1f
        .long    5
0:
        .string  "GNU"
1:
        .align 8
        .long    0xc0000002
        .long    3f - 2f
2:
        .long    0x3
3:
        .align 8
4:
