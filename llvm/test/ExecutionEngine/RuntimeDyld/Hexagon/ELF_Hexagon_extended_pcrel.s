# RUN: rm -rf %t && mkdir -p %t
# RUN: llvm-mc -triple=hexagon-unknown-elf -filetype=obj -o %t/ELF_Hexagon_extended_pcrel.o %s
# RUN: llvm-rtdyld -triple=hexagon-unknown-elf -verify -check=%s \
# RUN:   -map-section ELF_Hexagon_extended_pcrel.o,.text=0x1000 \
# RUN:   %t/ELF_Hexagon_extended_pcrel.o

# An extended branch uses two relocations. The extender carries bits 31:6,
# and the branch relocation carries only bits 5:0. In particular, the latter
# must not scatter the whole 22-bit displacement into the branch instruction.
	.text
	.globl main
main:
extended_call:
	{ call ##target }
	.space 0x100
	.p2align 2
	.globl target
target:
	{ jumpr r31 }

# rtdyld-check: (*{4}(extended_call))[13:0] = (target - extended_call)[19:6]
# rtdyld-check: (*{4}(extended_call))[27:16] = (target - extended_call)[31:20]
# The low relocation contains only the low six bits. The remaining immediate
# positions in the call word must remain zero.
# rtdyld-check: (*{4}(extended_call + 4))[6:1] = (target - extended_call)[5:0]
# rtdyld-check: (*{4}(extended_call + 4))[13:7] = 0
# rtdyld-check: (*{4}(extended_call + 4))[24:16] = 0
