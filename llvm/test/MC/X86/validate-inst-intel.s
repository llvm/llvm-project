# RUN: not llvm-mc -x86-asm-syntax intel -triple i686 -filetype asm -o /dev/null %s 2>&1 \
# RUN:    | FileCheck %s

	.text
	int 65535
# CHECK: error: invalid operand for instruction
# CHECK:	int 65535
# CHECK:            ^

	.text
	int -129
# CHECK: error: invalid operand for instruction
# CHECK:	int -129
# CHECK:            ^

