# RUN: not llvm-mc -triple mips-unknown-linux < %s -show-encoding -target-abi=o32 \
# RUN:     2>&1 | FileCheck %s
	.text
foo:
	.reloc 0, R_MIPS_32, .text+.text # CHECK: :[[@LINE]]:23: error: expression must be relocatable
	nop
