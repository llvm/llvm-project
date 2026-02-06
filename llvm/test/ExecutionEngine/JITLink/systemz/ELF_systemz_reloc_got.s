# REQUIRES: system-linux
# RUN: rm -rf %t && mkdir -p %t
# RUN: llvm-mc -triple=systemz-unknown-linux -position-independent \
# RUN:     -filetype=obj -o %t/elf_reloc.o %s
#
# RUN: llvm-jitlink -noexec \
# RUN:    -slab-allocate 100Kb -slab-address 0x6ff00000 -slab-page-size 4096 \
# RUN:    -abs foo=0x6ff04040 \
# RUN:    -abs bar=0x6ff04048 \
# RUN:     %t/elf_reloc.o -check %s

# Verifying GOT related relocations.

        .text
        .globl  main
        .type   main,@function
main:
# jitlink-check: decode_operand(main, 1) = _GLOBAL_OFFSET_TABLE_ - main 
	larl %r12, _GLOBAL_OFFSET_TABLE_
	.globl test_gotent_foo
test_gotent_foo:
# jitlink-check: decode_operand(test_gotent_foo, 1) = \
# jitlink-check:          (got_addr(elf_reloc.o, foo) - test_gotent_foo)
	.reloc .+2, R_390_GOTENT, foo+2
	larl %r1, 0
	.size test_gotent_foo, .-test_gotent_foo

	.globl test_gotent_bar
test_gotent_bar:
# jitlink-check: decode_operand(test_gotent_bar, 1) = \
# jitlink-check:          (got_addr(elf_reloc.o, bar) - test_gotent_bar) 
	.reloc .+2, R_390_GOTENT, bar+2
	larl %r1, 0
        .size test_gotent_bar, .-test_gotent_bar

        .globl test_gotpltent_foo
test_gotpltent_foo:
# jitlink-check: decode_operand(test_gotpltent_foo, 1) = \
# jitlink-check:          (got_addr(elf_reloc.o, foo) - test_gotpltent_foo)
	.reloc .+2, R_390_GOTPLTENT, foo+2
	larl %r1, 0
	.size test_gotpltent_foo, .-test_gotpltent_foo

        .globl test_gotpltent_bar
test_gotpltent_bar:
# jitlink-check: decode_operand(test_gotpltent_bar, 1) = \
# jitlink-check:          (got_addr(elf_reloc.o, bar) - test_gotpltent_bar)
	.reloc .+2, R_390_GOTPLTENT, bar+2
	larl %r1, 0
        .size test_gotpltent_bar, .-test_gotpltent_bar

       .globl test_got12_foo
test_got12_foo:
# jitlink-check: decode_operand(test_got12_foo, 2) = \
# jitlink-check:       (got_addr(elf_reloc.o, foo) - _GLOBAL_OFFSET_TABLE_)
	.reloc .+2, R_390_GOT12, foo
	l %r1, 0(%r12)
        .size test_got12_foo, .-test_got12_foo

      .globl test_got12_bar
test_got12_bar:
# jitlink-check: decode_operand(test_got12_bar, 2) = \
# jitlink-check:       (got_addr(elf_reloc.o, bar) - _GLOBAL_OFFSET_TABLE_)
	.reloc .+2, R_390_GOT12, bar
	l %r1, 0(%r12)
        .size test_got12_bar, .-test_got12_bar

       .globl test_gotplt12_foo
test_gotplt12_foo:
# jitlink-check: decode_operand(test_gotplt12_foo, 2) = \
# jitlink-check:       (got_addr(elf_reloc.o, foo) - _GLOBAL_OFFSET_TABLE_)
	.reloc .+2, R_390_GOTPLT12, foo
	l %r1, 0(%r12)
        .size test_gotplt12_foo, .-test_gotplt12_foo

       .globl test_gotplt12_bar
test_gotplt12_bar:
# jitlink-check: decode_operand(test_gotplt12_bar, 2) = \
# jitlink-check:       (got_addr(elf_reloc.o, bar) - _GLOBAL_OFFSET_TABLE_)
        .reloc .+2, R_390_GOTPLT12, bar 
        l %r1, 0(%r12)
        .size test_gotplt12_bar, .-test_gotplt12_bar

       .globl test_got20_foo
test_got20_foo:
# jitlink-check: decode_operand(test_got20_foo, 2) = \
# jitlink-check:       (got_addr(elf_reloc.o, foo) - _GLOBAL_OFFSET_TABLE_)
        .reloc .+2, R_390_GOT20, foo
        lg %r1, 0(%r12)
        .size test_got20_foo, .-test_got20_foo

      .globl test_got20_bar
test_got20_bar:
# jitlink-check: decode_operand(test_got20_bar, 2) = \
# jitlink-check:       (got_addr(elf_reloc.o, bar) - _GLOBAL_OFFSET_TABLE_)
        .reloc .+2, R_390_GOT20, bar
        lg %r1, 0(%r12)
        .size test_got20_bar, .-test_got20_bar

       .globl test_gotplt20_foo
test_gotplt20_foo:
# jitlink-check: decode_operand(test_gotplt20_foo, 2) = \
# jitlink-check:       (got_addr(elf_reloc.o, foo) - _GLOBAL_OFFSET_TABLE_)
        .reloc .+2, R_390_GOTPLT20, foo
        lg %r1, 0(%r12)
        .size test_gotplt20_foo, .-test_gotplt20_foo

       .globl test_gotplt20_bar
test_gotplt20_bar:
# jitlink-check: decode_operand(test_gotplt20_bar, 2) = \
# jitlink-check:       (got_addr(elf_reloc.o, bar) - _GLOBAL_OFFSET_TABLE_)
        .reloc .+2, R_390_GOTPLT20, bar
        lg %r1, 0(%r12)
        .size test_gotplt20_bar, .-test_gotplt20_bar
        br  %r14
	.size   main, .-main

	.data
	.globl test_got16_foo
# jitlink-check: *{2}test_got16_foo = \
# jitlink-check:     (got_addr(elf_reloc.o, foo) - _GLOBAL_OFFSET_TABLE_)
test_got16_foo:
	.reloc ., R_390_GOT16, foo
	.space 2
	.size test_got16_foo, .-test_got16_foo

       .globl test_got16_bar
# jitlink-check: *{2}test_got16_bar = \
# jitlink-check:     (got_addr(elf_reloc.o, bar) - _GLOBAL_OFFSET_TABLE_)
test_got16_bar:
        .reloc ., R_390_GOT16, bar
        .space 2
        .size test_got16_bar, .-test_got16_bar

        .globl test_gotplt16_foo
# jitlink-check: *{2}test_gotplt16_foo = \
# jitlink-check:    (got_addr(elf_reloc.o, foo) - _GLOBAL_OFFSET_TABLE_)
test_gotplt16_foo:
        .reloc ., R_390_GOTPLT16, foo
        .space 2
        .size test_gotplt16_foo, .-test_gotplt16_foo

       .globl test_gotplt16_bar
# jitlink-check: *{2}test_gotplt16_bar = \
# jitlink-check:    (got_addr(elf_reloc.o, bar) - _GLOBAL_OFFSET_TABLE_)
test_gotplt16_bar:
        .reloc ., R_390_GOTPLT16, bar
        .space 2
        .size test_gotplt16_bar, .-test_gotplt16_bar

        .globl test_got32_foo
# jitlink-check: *{4}test_got32_foo = \
# jitlink-check:    (got_addr(elf_reloc.o, foo) - _GLOBAL_OFFSET_TABLE_)
test_got32_foo:
        .reloc ., R_390_GOT32, foo
        .space 4 
        .size test_got32_foo, .-test_got32_foo

       .globl test_got32_bar
# jitlink-check: *{4}test_got32_bar = \
# jitlink-check:    (got_addr(elf_reloc.o, bar) - _GLOBAL_OFFSET_TABLE_)
test_got32_bar:
        .reloc ., R_390_GOT32, bar
        .space 4 
        .size test_got32_bar, .-test_got32_bar

        .globl test_gotplt32_foo
# jitlink-check: *{4}test_gotplt32_foo = \
# jitlink-check:    (got_addr(elf_reloc.o, foo) - _GLOBAL_OFFSET_TABLE_)
test_gotplt32_foo:
        .reloc ., R_390_GOTPLT32, foo
        .space 4 
        .size test_gotplt32_foo, .-test_gotplt32_foo

       .globl test_gotplt32_bar
# jitlink-check: *{4}test_gotplt32_bar = \
# jitlink-check:    (got_addr(elf_reloc.o, bar) - _GLOBAL_OFFSET_TABLE_)
test_gotplt32_bar:
        .reloc ., R_390_GOTPLT32, bar
        .space 4 
        .size test_gotplt32_bar, .-test_gotplt32_bar

        .globl test_got64_foo
# jitlink-check: *{8}test_got64_foo = \
# jitlink-check:    (got_addr(elf_reloc.o, foo) - _GLOBAL_OFFSET_TABLE_)
test_got64_foo:
        .reloc ., R_390_GOT64, foo
        .space 8 
        .size test_got64_foo, .-test_got64_foo

       .globl test_got64_bar
# jitlink-check: *{8}test_got64_bar = \
# jitlink-check:    (got_addr(elf_reloc.o, bar) - _GLOBAL_OFFSET_TABLE_)
test_got64_bar:
        .reloc ., R_390_GOT64, bar
        .space 8 
        .size test_got64_bar, .-test_got64_bar

        .globl test_gotplt64_foo
# jitlink-check: *{8}test_gotplt64_foo = \
# jitlink-check:    (got_addr(elf_reloc.o, foo) - _GLOBAL_OFFSET_TABLE_)
test_gotplt64_foo:
        .reloc ., R_390_GOTPLT64, foo
        .space 8 
        .size test_gotplt64_foo, .-test_gotplt64_foo

       .globl test_gotplt64_bar
# jitlink-check: *{8}test_gotplt64_bar = \
# jitlink-check:  (got_addr(elf_reloc.o, bar) - _GLOBAL_OFFSET_TABLE_)
test_gotplt64_bar:
        .reloc ., R_390_GOTPLT64, bar
        .space 8 
        .size test_gotplt64_bar, .-test_gotplt64_bar

        .globl test_gotpc_foo
# jitlink-check: *{4}test_gotpc_foo = _GLOBAL_OFFSET_TABLE_ - test_gotpc_foo
test_gotpc_foo:
        .reloc ., R_390_GOTPC, foo
        .space 4 
        .size test_gotpc_foo, .-test_gotpc_foo

        .globl test_gotpc_bar
# jitlink-check: *{4}test_gotpc_bar = _GLOBAL_OFFSET_TABLE_ - test_gotpc_bar
test_gotpc_bar:
        .reloc ., R_390_GOTPC, bar 
        .space 4
        .size test_gotpc_bar, .-test_gotpc_bar

        .globl test_gotpcdbl_foo
# jitlink-check: *{4}test_gotpcdbl_foo = \
# jitlink-check:           (_GLOBAL_OFFSET_TABLE_ - test_gotpcdbl_foo) >> 1
test_gotpcdbl_foo:
        .reloc ., R_390_GOTPCDBL, foo
        .space 4
        .size test_gotpcdbl_foo, .-test_gotpcdbl_foo

        .globl test_gotpcdbl_bar
# jitlink-check: *{4}test_gotpcdbl_bar =  \
# jitlink-check:           (_GLOBAL_OFFSET_TABLE_ - test_gotpcdbl_bar) >> 1
test_gotpcdbl_bar:
        .reloc ., R_390_GOTPCDBL, bar
        .space 4
        .size test_gotpcdbl_bar, .-test_gotpcdbl_bar

