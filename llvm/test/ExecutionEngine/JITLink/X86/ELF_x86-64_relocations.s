# RUN: rm -rf %t && mkdir -p %t
# RUN: llvm-mc -triple=x86_64-unknown-linux -position-independent -filetype=obj -o %t/elf_reloc.o %s
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
        .type   test_pcrel32,@function
test_pcrel32:
        movl    named_data(%rip), %eax
.Lend_test_pcrel32:
         .size   test_pcrel32, .Lend_test_pcrel32-test_pcrel32

# Test GOTPCREL handling. We want to check both the offset to the GOT entry and its
# contents.
# jitlink-check: decode_operand(test_gotpcrel, 4) = got_addr(elf_reloc.o, named_data) - next_pc(test_gotpcrel)
# jitlink-check: *{8}(got_addr(elf_reloc.o, named_data)) = named_data

        .globl test_gotpcrel
        .p2align      4, 0x90
        .type   test_gotpcrel,@function
test_gotpcrel:
	movl    named_data@GOTPCREL(%rip), %eax
.Lend_test_gotpcrel:
        .size   test_gotpcrel, .Lend_test_gotpcrel-test_gotpcrel

        .type   named_data,@object
        .data
        .p2align        2
named_data:
        .long   42
        .size   named_data, 4

        .ident  "clang version 10.0.0-4ubuntu1 "
        .section        ".note.GNU-stack","",@progbits
        .addrsig