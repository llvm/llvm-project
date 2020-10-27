# RUN: not llvm-mc -triple mips-unknown-linux < %s -show-encoding \
# RUN:     -target-abi=o32 2>&1 | FileCheck %s
	.text
foo:
	.reloc 0, R_MIPS_32, .text+.text  # CHECK: :[[@LINE]]:23: error: expression must be relocatable
	.reloc 0, 0, R_MIPS_32, .text     # CHECK: :[[@LINE]]:12: error: expected relocation name
	nop
