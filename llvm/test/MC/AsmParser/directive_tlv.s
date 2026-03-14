# RUN: llvm-mc -triple x86_64-unknown-darwin %s | FileCheck %s

# CHECK:	__DATA,__thread_vars,thread_local_variables
# CHECK:	.globl _a
# CHECK:	_a:
# CHECK:	.quad 0

	.tlv
.globl _a
_a:
	.quad 0
	.quad 0
	.quad 0
