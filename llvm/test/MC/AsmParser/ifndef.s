# RUN: llvm-mc -triple i386-unknown-unknown %s | FileCheck %s

# CHECK: .byte 1
# CHECK-NOT: byte 0
.ifndef undefined
	.byte 1
.else
	.byte 0
.endif

defined:

# CHECK-NOT: byte 0
# CHECK: .byte 1
.ifndef defined
	.byte 0
.else
	.byte 1
.endif

	movl	%eax, undefined

# CHECK: .byte 1
# CHECK-NOT: byte 0
.ifndef undefined
	.byte 1
.else
	.byte 0
.endif
