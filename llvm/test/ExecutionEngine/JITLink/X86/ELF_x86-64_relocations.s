# RUN: rm -rf %t && mkdir -p %t
# RUN: llvm-mc -triple=x86_64-unknown-linux -filetype=obj -o %t/elf_reloc.o %s
# RUN: llvm-jitlink -noexec -check %s %t/elf_reloc.o
#
# Test standard ELF relocations.

        .text
        .file   "testcase.c"

# Empty main entry point.
        .globl  main
        .p2align        4, 0x90
        .type   main,@function
main:
        retq
.Lfunc_end0:
        .size   main, .Lfunc_end0-main

# Test PCRel32 / R_X86_64_PC32 handling.
# jitlink-check: decode_operand(test_pcrel32, 4) = named_data - next_pc(test_pcrel32)
        .globl  test_pcrel32
        .p2align       4, 0x90
        .type  test_pcrel32,@function
test_pcrel32:
        movl    named_data(%rip), %eax
.Ltest_pcrel32_end:
        .size   test_pcrel32, .Ltest_pcrel32_end-test_pcrel32

        .type   named_data,@object
        .data
	.globl named_data
        .p2align        2
named_data:
        .long   42
        .size   named_data, 4

        .ident  "clang version 10.0.0-4ubuntu1 "
        .section        ".note.GNU-stack","",@progbits
        .addrsig
