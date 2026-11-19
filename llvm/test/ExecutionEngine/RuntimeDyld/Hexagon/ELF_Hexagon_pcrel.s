# RUN: rm -rf %t && mkdir -p %t
# RUN: llvm-mc -triple=hexagon-unknown-elf -filetype=obj -o %t/ELF_Hexagon_pcrel.o %s
# RUN: llvm-rtdyld -triple=hexagon-unknown-elf -verify \
# RUN:   -map-section ELF_Hexagon_pcrel.o,.text=0x1000 -check=%s %t/ELF_Hexagon_pcrel.o
# RUN: llvm-rtdyld -triple=hexagon-unknown-elf -verify \
# RUN:   -map-section ELF_Hexagon_pcrel.o,.text=0x7ffff000 -check=%s %t/ELF_Hexagon_pcrel.o

# Exercises resolveHexagonRelocation's R_HEX_B22_PCREL handling. The call's
# immediate must encode the displacement to target (here, one packet / 4
# bytes), not target's absolute load address, so the checked-in instruction
# word must come out identical however the section gets relocated.

	.text
	.globl main
main:
	{ call target }

	.p2align 2
	.globl target
target:
	{ jumpr r31 }

# rtdyld-check: *{4}(main) = 0x5a00c002
