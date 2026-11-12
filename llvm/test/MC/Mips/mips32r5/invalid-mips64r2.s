# Instructions that are invalid
#
# RUN: not llvm-mc %s -triple=mips64-unknown-linux -show-encoding \
# RUN:     -mcpu=mips32r5 2>%t1
# RUN: FileCheck %s < %t1

	.set noat
	dsbh	$v1,$t6    # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
	dshd	$v0,$sp    # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled

