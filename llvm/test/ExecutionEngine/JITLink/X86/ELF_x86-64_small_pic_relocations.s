# RUN: rm -rf %t && mkdir -p %t
# RUN: llvm-mc -triple=x86_64-unknown-linux -position-independent -filetype=obj \
# RUN:         -o %t/elf_sm_pic_reloc.o %s
# RUN: llvm-jitlink -noexec -slab-allocate 100Kb -slab-address 0xfff00000 \
# RUN:              -define-abs external_data=0x1 \
# RUN:              -define-abs extern_in_range32=0xffe00000 \
# RUN:              -define-abs extern_out_of_range32=0x7fff00000000 \
# RUN:              -check %s %t/elf_sm_pic_reloc.o
#
# Test ELF small/PIC relocations.

        .text
        .file   "testcase.c"

# Empty main entry point.
        .globl  main
        .p2align        4, 0x90
        .type   main,@function
main:
        retq

        .size   main, .-main

# Test PCRel32 / R_X86_64_PC32 handling.
# jitlink-check: decode_operand(test_pcrel32, 4) = named_data - next_pc(test_pcrel32)
        .globl  test_pcrel32
        .p2align       4, 0x90
        .type   test_pcrel32,@function
test_pcrel32:
        movl    named_data(%rip), %eax

         .size   test_pcrel32, .-test_pcrel32

        .globl  named_func
        .p2align       4, 0x90
        .type   named_func,@function
named_func:
        xorq    %rax, %rax

        .size   named_func, .-named_func

# Check R_X86_64_PLT32 handling with a call to a local function. This produces a
# Branch32 edge that is resolved like a regular PCRel32 (no PLT entry created).
#
# jitlink-check: decode_operand(test_call_local, 0) = named_func - next_pc(test_call_local)
        .globl  test_call_local
        .p2align       4, 0x90
        .type   test_call_local,@function
test_call_local:
        callq   named_func

        .size   test_call_local, .-test_call_local

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

# Check R_X86_64_PLT32 handling with a call to an external via PLT. This
# produces a Branch32ToStub edge, because externals are not defined locally.
# As the target is out-of-range from the callsite, the edge keeps using its PLT
# entry.
#
# jitlink-check: decode_operand(test_call_extern_plt, 0) = \
# jitlink-check:     stub_addr(elf_sm_pic_reloc.o, extern_out_of_range32) - \
# jitlink-check:        next_pc(test_call_extern_plt)
# jitlink-check: *{8}(got_addr(elf_sm_pic_reloc.o, extern_out_of_range32)) = \
# jitlink-check:     extern_out_of_range32
        .globl  test_call_extern_plt
        .p2align       4, 0x90
        .type   test_call_extern_plt,@function
test_call_extern_plt:
        callq   extern_out_of_range32@plt

        .size   test_call_extern_plt, .-test_call_extern_plt

# Test GOTPCREL handling. We want to check both the offset to the GOT entry and its
# contents.
# jitlink-check: decode_operand(test_gotpcrel, 4) = \
# jitlink-check:     got_addr(elf_sm_pic_reloc.o, named_data) - next_pc(test_gotpcrel)
# jitlink-check: *{8}(got_addr(elf_sm_pic_reloc.o, named_data)) = named_data

        .globl test_gotpcrel
        .p2align      4, 0x90
        .type   test_gotpcrel,@function
test_gotpcrel:
	movl    named_data@GOTPCREL(%rip), %eax

        .size   test_gotpcrel, .-test_gotpcrel

# Test REX_GOTPCRELX handling. We want to check both the offset to the GOT entry and its
# contents.
# jitlink-check: decode_operand(test_rex_gotpcrelx, 4) = \
# jitlink-check:   got_addr(elf_sm_pic_reloc.o, external_data) - next_pc(test_rex_gotpcrelx)

        .globl test_rex_gotpcrelx
        .p2align      4, 0x90
        .type   test_rex_gotpcrelx,@function
test_rex_gotpcrelx:
	movq    external_data@GOTPCREL(%rip), %rax

        .size   test_rex_gotpcrelx, .-test_rex_gotpcrelx

# Test GOTOFF64 handling.
# jitlink-check: decode_operand(test_gotoff64, 1) = named_func - _GLOBAL_OFFSET_TABLE_
        .globl test_gotoff64
        .p2align     4, 0x90
        .type  test_gotoff64,@function
test_gotoff64:
        movabsq $named_func@GOTOFF, %rax

        .size   test_gotoff64, .-test_gotoff64

# Test that relocations to anonymous constant pool entries work.
        .globl  test_anchor_LCPI
        .p2align        4, 0x90
        .type   test_anchor_LCPI,@function
test_anchor_LCPI:
        movq    .LCPI0_0(%rip), %rax

        .size   test_anchor_LCPI, .-test_anchor_LCPI

        .data

        .type   named_data,@object
        .p2align        3
named_data:
        .quad   42
        .size   named_data, 8

# Test BSS / zero-fill section handling.
# llvm-jitlink: *{4}bss_variable = 0

	.type	bss_variable,@object
	.bss
	.globl	bss_variable
	.p2align	2
bss_variable:
	.long	0
	.size	bss_variable, 4

# Constant pool entry with type STT_NOTYPE.
        .section        .rodata.cst8,"aM",@progbits,8
        .p2align        3
.LCPI0_0:
        .quad   0x400921fb54442d18

        .ident  "clang version 10.0.0-4ubuntu1 "
        .section        ".note.GNU-stack","",@progbits
        .addrsig
