# RUN: llvm-mc -arch=hexagon -filetype=asm %s 2> %t; FileCheck %s < %t

# Check that changes to a read-only register is caught.

	{ pc = r0 }
# CHECK: error: Cannot write to read-only register
