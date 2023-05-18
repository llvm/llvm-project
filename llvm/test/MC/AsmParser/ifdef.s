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

# .ifdef directive does not count as a use, so ensure redefinition is permitted
.set var, 1
.ifdef var
	.set var, 2
.else
	.set var, 3
.endif
# CHECK: .byte 2
.byte var

.set var, 4
# CHECK: .byte 4
.byte var
