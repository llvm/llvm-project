# RUN: not llvm-mc -triple s390x-unknown-unknown -filetype=obj %s 2>&1 | FileCheck %s

	.text

# CHECK: error: displacement exceeds uint12
	la    %r1, b-a(%r1)

# CHECK: error: displacement exceeds int20
        lay   %r1, d-c(%r1)

# CHECK-NOT: error
        lay   %r1, b-a(%r1)

	.type	a,@object
	.local	a
	.comm	a,4096
	.type	b,@object
	.local	b
	.comm	b,4,4

	.type	c,@object
	.local	c
	.comm	c,524288
	.type	d,@object
	.local	d
	.comm	d,4,4

