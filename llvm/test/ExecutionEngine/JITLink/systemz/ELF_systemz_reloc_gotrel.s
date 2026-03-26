# REQUIRES: system-linux
# RUN: rm -rf %t && mkdir -p %t
# RUN: llvm-mc -triple=systemz-unknown-linux -position-independent \
# RUN:     -filetype=obj -o %t/elf_reloc.o %s
#
# RUN: llvm-jitlink -noexec \
# RUN:    -slab-allocate 100Kb -slab-address 0x6ff00000 -slab-page-size 4096 \
# RUN:    -abs foo=0x6ff04080 \
# RUN:    -abs bar=0x6ff04040 \
# RUN:     %t/elf_reloc.o -check %s

        .text
        .globl  main
        .type   main,@function
main:
        br  %r14
	.size   main, .-main

	.data
	.globl test_gotoff16_bar
# jitlink-check: *{2}test_gotoff16_bar = (bar - _GLOBAL_OFFSET_TABLE_) & 0xffff
test_gotoff16_bar:
	.reloc ., R_390_GOTOFF16, bar
	.space 2
	.size test_gotoff16_bar, .-test_gotoff16_bar

       .globl test_pltoff16_foo
# jitlink-check: *{2}test_pltoff16_foo =  \
# jitlink-check:      (stub_addr(elf_reloc.o, foo) - _GLOBAL_OFFSET_TABLE_) \
# jitlink-check:       & 0xffff
test_pltoff16_foo:
        .reloc ., R_390_PLTOFF16, foo 
        .space 2
        .size test_pltoff16_foo, .-test_pltoff16_foo


       .globl test_gotoff32_bar
# jitlink-check: *{4}test_gotoff32_bar = (bar - _GLOBAL_OFFSET_TABLE_) \
# jitlink-check:       & 0xffffffff
test_gotoff32_bar:
        .reloc ., R_390_GOTOFF, bar
        .space 4 
        .size test_gotoff32_bar, .-test_gotoff32_bar

        .globl test_pltoff32_foo
# jitlink-check: *{4}test_pltoff32_foo = \
# jitlink-check:      (stub_addr(elf_reloc.o, foo) - _GLOBAL_OFFSET_TABLE_) \
# jitlink-check:       & 0xffffffff
test_pltoff32_foo:
        .reloc ., R_390_PLTOFF32, foo
        .space 4 
        .size test_pltoff32_foo, .-test_pltoff32_foo

	 .globl test_gotoff64_bar
# jitlink-check: *{8}test_gotoff64_bar = bar - _GLOBAL_OFFSET_TABLE_
test_gotoff64_bar:
        .reloc ., R_390_GOTOFF64, bar
        .space 8 
        .size test_gotoff64_bar, .-test_gotoff64_bar

        .globl test_pltoff64_foo
# jitlink-check: *{8}test_pltoff64_foo = \
# jitlink-check:      (stub_addr(elf_reloc.o, foo) - _GLOBAL_OFFSET_TABLE_)
test_pltoff64_foo:
        .reloc ., R_390_PLTOFF64, foo
        .space 8 
        .size test_pltoff64_foo, .-test_pltoff64_foo

