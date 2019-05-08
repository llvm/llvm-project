# RUN: llvm-mc -triple -x86_64-unknown-unknown -filetype obj -o - %s | llvm-readobj -S --sd \
# RUN:   | FileCheck %s

	.text
	even_check:
	.byte 0x00
	.byte 0x01
	.byte 0x02
	.byte 0x03
	.byte 0x04
	.byte 0x05
	.byte 0x06
	.byte 0x07
	.byte 0x08
	.byte 0x09
	.byte 0x10
	.even
	.byte 0x11
	.byte 0x12
	.even
	.byte 0x13
	.even
	.byte 0x00
	.byte 0x01
	.byte 0x02
	.byte 0x03
	.byte 0x04
	.byte 0x05
	.byte 0x06
	.byte 0x07
	.byte 0x08
	.byte 0x09
	.byte 0x10
	.byte 0x11
	.byte 0x12
	.byte 0x13
	.byte 0x14
	.byte 0x15

# CHECK: Section {
# CHECK:   Name: .text
# CHECK:   SectionData (
# CHECK:      0000: 00010203 04050607 08091090 11121390
# CHECK:	  0010: 00010203 04050607 08091011 12131415
# CHECK:   )
# CHECK: }

