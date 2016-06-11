// RUN: not llvm-mc -triple aarch64-unknown-none-eabi -filetype asm -o - %s 2>&1 | FileCheck %s

	.arch axp64
# CHECK: error: unknown arch name
# CHECK: 	.arch axp64
# CHECK:	      ^

	.arch armv8

	fminnm d0, d0, d1

# CHECK: error: instruction requires: fp-armv8
# CHECK: 	fminnm d0, d0, d1
# CHECK:	^

	.arch armv8+fp

# CHECK: '+fp' is not a recognized feature for this target (ignoring feature)

	fminnm d0, d0, d1

# CHECK: error: instruction requires: fp-armv8
# CHECK: 	fminnm d0, d0, d1
# CHECK:	^

	.arch armv8+neon

	.arch armv8

	fminnm d0, d0, d1

# CHECK: error: instruction requires: fp-armv8
# CHECK: 	fminnm d0, d0, d1
# CHECK:	^

