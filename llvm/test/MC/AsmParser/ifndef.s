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

# .ifndef directive does not count as a use, so ensure redefinition is permitted
.set var, 1
.ifndef var
	.set var, 2
.else
	.set var, 3
.endif
# CHECK: .byte 3
.byte var

.set var, 4
# CHECK: .byte 4
.byte var
