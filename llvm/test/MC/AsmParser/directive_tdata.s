# RUN: llvm-mc -triple x86_64-unknown-darwin %s | FileCheck %s

# CHECK:	__DATA,__thread_data,thread_local_regular
# CHECK:	_a$tlv$init:
# CHECK:	.quad 4

	.tdata
_a$tlv$init:
	.quad 4
