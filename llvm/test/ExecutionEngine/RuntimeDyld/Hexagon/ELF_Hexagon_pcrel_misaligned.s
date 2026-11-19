# RUN: rm -rf %t && mkdir -p %t
# RUN: llvm-mc -triple=hexagon-unknown-elf -filetype=obj -o %t/ELF_Hexagon_pcrel_misaligned.o %s
# RUN: not llvm-rtdyld -triple=hexagon-unknown-elf -verify -check=%s \
# RUN:   -map-section ELF_Hexagon_pcrel_misaligned.o,.text=0x1000 \
# RUN:   -map-section ELF_Hexagon_pcrel_misaligned.o,.data=0x2000 \
# RUN:   %t/ELF_Hexagon_pcrel_misaligned.o 2>&1 | FileCheck %s

# A call displacement is a word count, so a target that is not 4-byte aligned
# cannot be encoded.

	.text
	.p2align 2
	.globl main
main:
	{ call odd }

	.data
	.byte 0
	.globl odd
odd:
	.byte 0

# rtdyld-check: *{4}(main) = *{4}(main)

# CHECK: R_HEX_B22_PCREL relocation target 0x{{[0-9a-f]+}} is not 4-byte aligned
