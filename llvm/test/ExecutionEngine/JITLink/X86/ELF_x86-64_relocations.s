# RUN: rm -rf %t && mkdir -p %t
# RUN: llvm-mc -triple=x86_64-unknown-linux -filetype=obj -o %t/elf_reloc.o %s
# RUN: llvm-jitlink -noexec %t/elf_reloc.o
#
# Test standard ELF relocations.

        .text
        .file   "testcase.c"
        .globl  main
        .p2align        4, 0x90
        .type   main,@function
main:
        movl    $42, %eax
        retq
.Lfunc_end0:
        .size   main, .Lfunc_end0-main

        .ident  "clang version 10.0.0-4ubuntu1 "
        .section        ".note.GNU-stack","",@progbits
        .addrsig