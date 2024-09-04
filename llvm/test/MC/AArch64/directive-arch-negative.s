// RUN: not llvm-mc -triple aarch64 -filetype asm -o - %s 2>&1 | FileCheck %s

	.arch axp64
# CHECK: error: unknown arch name
# CHECK-NEXT: 	.arch axp64
# CHECK-NEXT:	      ^

	.arch armv8
	aese v0.8h, v1.8h

# CHECK: error: invalid operand for instruction
# CHECK-NEXT: 	aese v0.8h, v1.8h
# CHECK-NEXT:	^

	.arch armv8+foo+nobar
	aese v0.8h, v1.8h

# CHECK: error: unsupported architectural extension: foo
# CHECK-NEXT:   .arch armv8+foo+nobar
# CHECK-NEXT:               ^

# CHECK: error: invalid operand for instruction
# CHECK-NEXT:	aese v0.8h, v1.8h
# CHECK-NEXT:	^

	.arch armv8+crypto

	.arch armv8

	aese v0.8h, v1.8h

# CHECK: error: invalid operand for instruction
# CHECK-NEXT: 	aese v0.8h, v1.8h
# CHECK-NEXT:	^

	.arch armv8.1-a+noras
	esb

# CHECK: error: instruction requires: ras
# CHECK-NEXT:   esb

	.arch armv8
        casa  w5, w7, [x19]

# CHECK: error: instruction requires: lse
# CHECK-NEXT:   casa  w5, w7, [x19]

	.arch armv8+crypto
        crc32b w0, w1, w2

# CHECK: error: instruction requires: crc
# CHECK-NEXT:   crc32b w0, w1, w2

	.arch armv8.1-a+nolse
        casa  w5, w7, [x20]

# CHECK: error: instruction requires: lse
# CHECK-NEXT:   casa  w5, w7, [x20]
