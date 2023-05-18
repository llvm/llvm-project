# RUN: llvm-mc -triple i386-unknown-unknown %s | FileCheck %s

# CHECK-NOT: .byte 0
# CHECK: .byte 1
.ifnes "foo space", "foo space"
	.byte 0
.else
	.byte 1
.endif

# CHECK-NOT: .byte 0
# CHECK: .byte 1
.ifnes "unequal", "unEqual"
	.byte 1
.else
	.byte 0
.endif

# CHECK-NOT: .byte 0
# CHECK: .byte 1
.ifnes "equal", "equal" ; .byte 0 ; .else ; .byte 1 ; .endif

