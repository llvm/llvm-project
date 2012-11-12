# RUN: llvm-mc -triple i386-unknown-unknown %s | FileCheck %s

# CHECK-NOT: .byte 0
# CHECK: .byte 1
.ifdef undefined
	.byte 0
.else
	.byte 1
.endif

defined:

# CHECK: .byte 1
# CHECK-NOT: .byte 0
.ifdef defined
	.byte 1
.else
	.byte 0
.endif

	movl	%eax, undefined

# CHECK-NOT: .byte 0
# CHECK: .byte 1
.ifdef undefined
	.byte 0
.else
	.byte 1
.endif
