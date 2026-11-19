# RUN: rm -rf %t && mkdir -p %t
# RUN: llvm-mc -triple=hexagon-unknown-elf -filetype=obj -o %t/ELF_Hexagon_pcrel_range.o %s
# RUN: llvm-rtdyld -triple=hexagon-unknown-elf -verify -check=%s \
# RUN:   -map-section ELF_Hexagon_pcrel_range.o,.text=0x1000000 \
# RUN:   -map-section ELF_Hexagon_pcrel_range.o,.far_pos=0x17ffffc \
# RUN:   -map-section ELF_Hexagon_pcrel_range.o,.far_neg=0x800004 \
# RUN:   %t/ELF_Hexagon_pcrel_range.o

# The R_HEX_B22_PCREL displacement is a signed 22-bit word count, so the
# reachable range is [-0x800000, 0x7ffffc] bytes. Check both extremes. The
# first call is at 0x1000000 and the second at 0x1000004.

	.text
	.p2align 2
	.globl main
main:
	{ call far_pos }
	{ call far_neg }

	.section .far_pos,"ax"
	.p2align 2
	.globl far_pos
far_pos:
	{ jumpr r31 }

	.section .far_neg,"ax"
	.p2align 2
	.globl far_neg
far_neg:
	{ jumpr r31 }

# far_pos - main = 0x7ffffc: the largest positive word count, 0x1fffff.
# rtdyld-check: *{4}(main) = 0x5afffffe
# far_neg - (main + 4) = -0x800000: only the sign bit set.
# rtdyld-check: *{4}(main + 4) = 0x5b00c000
