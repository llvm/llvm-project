# RUN: rm -rf %t && mkdir -p %t
# RUN: llvm-mc -triple=hexagon-unknown-elf -filetype=obj -o %t/ELF_Hexagon_pcrel_out_of_range.o %s
# RUN: not llvm-rtdyld -triple=hexagon-unknown-elf -verify -check=%s \
# RUN:   -map-section ELF_Hexagon_pcrel_out_of_range.o,.text=0x1000000 \
# RUN:   -map-section ELF_Hexagon_pcrel_out_of_range.o,.far=0x1800000 \
# RUN:   %t/ELF_Hexagon_pcrel_out_of_range.o 2>&1 | FileCheck %s --check-prefix=FWD
# RUN: not llvm-rtdyld -triple=hexagon-unknown-elf -verify -check=%s \
# RUN:   -map-section ELF_Hexagon_pcrel_out_of_range.o,.text=0x1000000 \
# RUN:   -map-section ELF_Hexagon_pcrel_out_of_range.o,.far=0x7ffffc \
# RUN:   %t/ELF_Hexagon_pcrel_out_of_range.o 2>&1 | FileCheck %s --check-prefix=BWD

# One word past either end of the R_HEX_B22_PCREL range must be reported
# as an error instead of being truncated into a wrong branch.

	.text
	.p2align 2
	.globl main
main:
	{ call far }

	.section .far,"ax"
	.p2align 2
	.globl far
far:
	{ jumpr r31 }

# llvm-rtdyld requires at least one check; the relocation error is reported
# after the checks have run.
# rtdyld-check: *{4}(main) = *{4}(main)

# FWD: R_HEX_B22_PCREL relocation out of range
# BWD: R_HEX_B22_PCREL relocation out of range
