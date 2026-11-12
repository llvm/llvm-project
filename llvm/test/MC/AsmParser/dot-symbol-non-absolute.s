# RUN: not llvm-mc -filetype=obj -triple i386-unknown-unknown %s 2> %t
# RUN: FileCheck -input-file %t %s


	.extern foo

# CHECK: : expected absolute expression
. = foo + 10
	.byte 1
