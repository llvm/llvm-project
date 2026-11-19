# RUN: rm -rf %t && mkdir -p %t
# RUN: llvm-mc -triple=hexagon-unknown-elf -filetype=obj -o %t/ELF_Hexagon_data_relocs.o %s
# RUN: llvm-rtdyld -triple=hexagon-unknown-elf -verify -check=%s \
# RUN:   -map-section ELF_Hexagon_data_relocs.o,.text=0x1000 \
# RUN:   -map-section ELF_Hexagon_data_relocs.o,.data8=0x2000 \
# RUN:   -map-section ELF_Hexagon_data_relocs.o,.data16=0x3000 \
# RUN:   -map-section ELF_Hexagon_data_relocs.o,.data32=0x4000 \
# RUN:   %t/ELF_Hexagon_data_relocs.o

# R_HEX_8 and R_HEX_16 must touch only their own one and two bytes. The
# relocations below are the last bytes of their sections, so reading or
# writing a whole instruction word would run past the end of the section, and
# the bytes before them must stay unchanged.

	.text
	.p2align 2
	.globl main
main:
	{ jumpr r31 }
	.globl target
target:
	{ jumpr r31 }

	.section .data8,"aw"
first8:
	.byte 0xaa
	.byte 0xbb
	.byte 0xcc
last8:
	.byte target

	.section .data16,"aw"
first16:
	.short 0x1122
	.short 0x3344
last16:
	.short target

# R_HEX_32 has no alignment requirement, so it may refer to an odd address.
	.section .data32,"aw"
	.p2align 2
word:
	.word last8

# target is at 0x1004.
# rtdyld-check: *{1}(first8) = 0xaa
# rtdyld-check: *{1}(first8 + 1) = 0xbb
# rtdyld-check: *{1}(first8 + 2) = 0xcc
# rtdyld-check: *{1}(last8) = 0x04
# rtdyld-check: *{2}(first16) = 0x1122
# rtdyld-check: *{2}(first16 + 2) = 0x3344
# rtdyld-check: *{2}(last16) = 0x1004
# last8 is at 0x2003.
# rtdyld-check: *{4}(word) = 0x2003
