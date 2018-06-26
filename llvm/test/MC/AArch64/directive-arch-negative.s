// RUN: not llvm-mc -triple aarch64-unknown-none-eabi -filetype asm -o - %s 2>&1 | FileCheck %s

	.arch axp64
# CHECK: error: unknown arch name
# CHECK-NEXT: 	.arch axp64
# CHECK-NEXT:	      ^

	.arch armv8
	aese v0.8h, v1.8h

# CHECK: error: invalid operand for instruction
# CHECK-NEXT: 	aese v0.8h, v1.8h
# CHECK-NEXT:	^

// We silently ignore invalid features.
	.arch armv8+foo
	aese v0.8h, v1.8h

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

// PR32873: without extra features, '.arch' is currently ignored.
// Add an unrelated feature to accept the directive.
	.arch armv8+crc
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
