# RUN: llvm-mc -triple powerpc64le-unknown-unknown %s 2>&1 | FileCheck %s

# Test that CFI directives can handle registers with a '%' prefix.

# CHECK-LABEL:	__test1
# CHECK:	.cfi_startproc
# CHECK-NEXT:   mflr	12
# CHECK-NEXT:   .cfi_register lr, r12

	.globl __test1
__test1:
	.cfi_startproc
	mflr %r12
	.cfi_register lr,%r12
	blr
	.cfi_endproc
