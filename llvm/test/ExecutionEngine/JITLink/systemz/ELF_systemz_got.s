# REQUIRES: system-linux
# RUN: rm -rf %t && mkdir -p %t
# RUN: llvm-mc -triple=systemz-unknown-linux -position-independent \
# RUN:     -filetype=obj -o %t/elf_reloc.o %s
#
# RUN: llvm-jitlink -noexec \
# RUN:    -slab-allocate 100Kb -slab-address 0x6ff00000 -slab-page-size 4096 \
# RUN:    -abs foo=0x6ff04040 \
# RUN:    -abs bar=0x6ff04048 \
# RUN:     %t/elf_reloc.o
#
# Check R_390_GOT* handling.

        .text
        .globl  main
        .type   main,@function
main:
	larl %r12, _GLOBAL_OFFSET_TABLE_
	.reloc .+2, R_390_GOTENT, foo+2
	larl %r1, 0
	.reloc .+2, R_390_GOTENT, bar+2
	larl %r1, 0
	.reloc .+2, R_390_GOTPLTENT, foo+2
	larl %r1, 0
	.reloc .+2, R_390_GOTPLTENT, bar+2
	larl %r1, 0
	.reloc .+2, R_390_GOT12, foo
	l %r1, 0(%r12)
	.reloc .+2, R_390_GOT12, bar
	l %r1, 0(%r12)
	.reloc .+2, R_390_GOTPLT12, foo
	l %r1, 0(%r12)
	.reloc .+2, R_390_GOTPLT12, bar
	l %r1, 0(%r12)
	.reloc .+2, R_390_GOT20, foo
	lg %r1, 0(%r12)
	.reloc .+2, R_390_GOT20, bar
	lg %r1, 0(%r12)
	.reloc .+2, R_390_GOTPLT20, foo
	lg %r1, 0(%r12)
	.reloc .+2, R_390_GOTPLT20, bar
	lg %r1, 0(%r12)
        br  %r14
	.size   main, .-main

	.data
	.reloc ., R_390_GOT16, foo
	.space 2
	.reloc ., R_390_GOT16, bar
	.space 2
	.reloc ., R_390_GOTPLT16, foo
	.space 2
	.reloc ., R_390_GOTPLT16, bar
	.space 2
	.reloc ., R_390_GOT32, foo
	.space 4
	.reloc ., R_390_GOT32, bar
	.space 4
	.reloc ., R_390_GOTPLT32, foo
	.space 4
	.reloc ., R_390_GOTPLT32, bar
	.space 4
	.reloc ., R_390_GOT64, foo
	.space 8
	.reloc ., R_390_GOT64, bar
	.space 8
	.reloc ., R_390_GOTPLT64, foo
	.space	8
	.reloc ., R_390_GOTPLT64, bar
	.space 8
	.reloc ., R_390_GOTPC, foo
        .space 4
	.reloc ., R_390_GOTPC, bar 
        .space 4
	.reloc ., R_390_GOTPCDBL, foo
        .space 4
	.reloc ., R_390_GOTPCDBL, bar 
        .space 4
