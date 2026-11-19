# RUN: rm -rf %t && mkdir -p %t
# RUN: llvm-mc -triple=hexagon-unknown-elf -filetype=obj -o %t/ELF_Hexagon_relocations.o %s
# RUN: llvm-rtdyld -triple=hexagon-unknown-elf -verify -check=%s %t/ELF_Hexagon_relocations.o

# Exercises RuntimeDyldELF's Hexagon support (resolveHexagonRelocation /
# HexagonFindMask's doReloc), which scatters a relocation's value into the
# non-contiguous bit positions the Hexagon ABI assigns to R_HEX_32.

	.text
	.globl main
main:
	.word target
	.p2align 2
	.globl target
target:
	{ jumpr r31 }

# rtdyld-check: *{4}(main) = target
