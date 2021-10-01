# RUN: rm -rf %t && mkdir -p %t
# RUN: llvm-mc -triple=x86_64-unknown-linux -position-independent \
# RUN:     -filetype=obj -o %t/elf_sm_pic_reloc.o %s
# RUN: llvm-jitlink -noexec \
# RUN:     -slab-allocate 100Kb -slab-address 0xfff00000 -slab-page-size 4096 \
# RUN:     -define-abs extern_in_range32=0xffe00000 \
# RUN:     -check %s %t/elf_sm_pic_reloc.o
#


        .text
        .file   "testcase.c"

# Empty main entry point.
        .globl  main
        .p2align        4, 0x90
        .type   main,@function
main:
        retq

        .size   main, .-main

# Test optimization of transforming "call *foo@GOTPCREL(%rip)" to "addr call foo"
# We need check both the target address and the instruction opcodes
# jitlink-check: decode_operand(test_call_gotpcrelx, 0)[31:0] = extern_in_range32
# jitlink-check: *{1}test_call_gotpcrelx = 0x67
# jitlink-check: *{1}test_call_gotpcrelx+1 = 0xe8
        .globl test_call_gotpcrelx
        .p2align      4, 0x90
        .type   test_call_gotpcrelx,@function
test_call_gotpcrelx:
	call    *extern_in_range32@GOTPCREL(%rip)

        .size   test_call_gotpcrelx, .-test_call_gotpcrelx


# Test optimization of transforming "jmp *foo@GOTPCREL(%rip)" to "jmp foo ; nop"
# We need check both the target address and the instruction opcodes
# jitlink-check: decode_operand(test_call_gotpcrelx, 0)[31:0] = extern_in_range32
# jitlink-check: *{1}test_jmp_gotpcrelx = 0xe9
# jitlink-check: *{1}test_jmp_gotpcrelx+5 = 0x90
        .globl test_jmp_gotpcrelx
        .p2align      4, 0x90
        .type   test_jmp_gotpcrelx,@function
test_jmp_gotpcrelx:
	jmp    *extern_in_range32@GOTPCREL(%rip)

        .size   test_jmp_gotpcrelx, .-test_jmp_gotpcrelx

# Check R_X86_64_PLT32 handling with a call to an external. This produces a
# Branch32ToStub edge, because externals are not defined locally. During
# resolution, the target turns out to be in-range from the callsite and so the
# edge is relaxed in post-allocation optimization.
#
# jitlink-check: decode_operand(test_call_extern, 0) = \
# jitlink-check:     extern_in_range32 - next_pc(test_call_extern)
        .globl  test_call_extern
        .p2align       4, 0x90
        .type   test_call_extern,@function
test_call_extern:
        callq   extern_in_range32@plt

        .size   test_call_extern, .-test_call_extern

