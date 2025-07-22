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

	.text
	loop WORD PTR [SYM+4]
# CHECK: error: invalid operand for instruction
# CHECK:	loop WORD PTR [SYM+4]
# CHECK:        ^

	.text
	loope BYTE PTR [128]
# CHECK: error: invalid operand for instruction
# CHECK:	loope BYTE PTR [128]
# CHECK:        ^

	.text
	loopne BYTE PTR [-129]
# CHECK: error: invalid operand for instruction
# CHECK:	loopne BYTE PTR [-129]
# CHECK:        ^

	.text
	jrcxz XMMWORD PTR [0]
# CHECK: error: invalid operand for instruction
# CHECK:	jrcxz XMMWORD PTR [0]
# CHECK:        ^

	.text
	jecxz BYTE PTR[-444]
# CHECK: error: invalid operand for instruction
# CHECK:	jecxz BYTE PTR[-444]
# CHECK:        ^

	.text
	jcxz BYTE PTR[444]
# CHECK: error: invalid operand for instruction
# CHECK:	jcxz BYTE PTR[444]
# CHECK:        ^

