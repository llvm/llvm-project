# RUN: not llvm-mc -triple s390x-unknown-unknown -filetype=obj %s 2>&1 | FileCheck %s

	.text

# CHECK:      error: operand out of range (4096 not between 0 and 4095)
# CHECK-NEXT:        la %r1, b-a(%r1)
# CHECK-NEXT:        ^
	la    %r1, b-a(%r1)

# CHECK-NEXT: error: operand out of range (524288 not between -524288 and 524287)
# CHECK-NEXT:        lay   %r1, d-c(%r1)
# CHECK-NEXT:        ^
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

