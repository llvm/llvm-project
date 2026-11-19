# RUN: rm -rf %t && mkdir -p %t
# RUN: llvm-mc -triple=hexagon-unknown-elf -filetype=obj -o %t/ELF_Hexagon_extended_pcrel_far.o %s
# RUN: llvm-rtdyld -triple=hexagon-unknown-elf -verify -check=%s \
# RUN:   -map-section ELF_Hexagon_extended_pcrel_far.o,.text=0x1000 \
# RUN:   -map-section ELF_Hexagon_extended_pcrel_far.o,.far=0x4001000 \
# RUN:   %t/ELF_Hexagon_extended_pcrel_far.o

# An extended call reaches beyond the +-8MB range of a plain call. The branch
# relocation applies only displacement bits 5:0, so its 22-bit ABI range must
# not be enforced on the full 0x4000000-byte displacement.

	.text
	.globl main
main:
extended_call:
	{ call ##far_target }

	.section .far,"ax"
	.p2align 2
	.globl far_target
far_target:
	{ jumpr r31 }

# rtdyld-check: (*{4}(extended_call))[13:0] = (far_target - extended_call)[19:6]
# rtdyld-check: (*{4}(extended_call))[27:16] = (far_target - extended_call)[31:20]
# rtdyld-check: (*{4}(extended_call + 4))[6:1] = (far_target - extended_call)[5:0]
