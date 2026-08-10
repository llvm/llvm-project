# Instructions that are invalid
#
# RUN: not llvm-mc %s -triple=mips-unknown-linux -show-encoding -mcpu=mips64 \
# RUN:     2>%t1
# RUN: FileCheck %s < %t1

	.set noat
        eretnc                      # CHECK: :[[@LINE]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
