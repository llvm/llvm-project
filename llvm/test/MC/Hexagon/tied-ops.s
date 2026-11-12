# RUN: llvm-mc -triple=hexagon -filetype=asm %s 2> %t; FileCheck %s < %t

# Check that tied operands are caught

	{ r0 = sub(##_start, asl(r1, #1)) }
# CHECK: error: invalid operand for instruction
