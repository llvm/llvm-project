# RUN: not llvm-mc -triple mips-unknown-linux %s -show-encoding \
# RUN:     -target-abi=o32 -filetype=obj 2>&1 | FileCheck %s
	.text
	nop
	.reloc foo, R_MIPS_32, .text  # CHECK: :[[@LINE]]:2: error: unresolved relocation offset
	nop
	nop
	.reloc bar, R_MIPS_32, .text  # CHECK: :[[@LINE]]:2: error: unresolved relocation offset
	nop
