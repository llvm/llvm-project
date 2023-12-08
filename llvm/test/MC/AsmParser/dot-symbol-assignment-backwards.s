# RUN: not llvm-mc -filetype=obj -triple i386-unknown-unknown %s 2> %t
# RUN: FileCheck -input-file %t %s

. = 0x10
	.byte 1

. = . + 10
	.byte 2

# CHECK: error: invalid .org offset '24' (at offset '28')
. = 0x18
	.byte 3
