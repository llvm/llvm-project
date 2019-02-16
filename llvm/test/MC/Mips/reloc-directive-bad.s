# RUN: not llvm-mc -triple mips-unknown-linux < %s -show-encoding \
# RUN:     -target-abi=o32 2>&1 | FileCheck %s
	.text
foo:
	.reloc foo+4, R_MIPS_32, .text    # CHECK: :[[@LINE]]:9: error: expected non-negative number or a label
	.reloc foo+foo, R_MIPS_32, .text  # CHECK: :[[@LINE]]:9: error: expected non-negative number or a label
	.reloc 0, R_MIPS_32, .text+.text  # CHECK: :[[@LINE]]:23: error: expression must be relocatable
	.reloc 0 R_MIPS_32, .text         # CHECK: :[[@LINE]]:11: error: expected comma
	.reloc 0, 0, R_MIPS_32, .text     # CHECK: :[[@LINE]]:12: error: expected relocation name
	.reloc -1, R_MIPS_32, .text       # CHECK: :[[@LINE]]:9: error: expression is negative
	.reloc 1b, R_MIPS_32, .text       # CHECK: :[[@LINE]]:9: error: directional label undefined
	.reloc 1f, R_MIPS_32, .text       # CHECK: :[[@LINE]]:9: error: directional label undefined
	nop
