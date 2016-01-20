# RUN: not llvm-mc -triple mips-unknown-linux < %s -show-encoding -target-abi=o32 \
# RUN:     2>&1 | FileCheck %s
	.text
foo:
	.reloc -1, R_MIPS_32, .text # CHECK: :[[@LINE]]:9: error: expression is negative
	nop
