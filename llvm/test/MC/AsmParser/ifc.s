# RUN: llvm-mc -triple i386-unknown-unknown %s | FileCheck %s

# CHECK-NOT: .byte 0
# CHECK: .byte 1
.ifc foo, foo
	.byte 1
.else
	.byte 0
.endif

# CHECK-NOT: .byte 0
# CHECK: .byte 1
.ifc "foo space", "foo space"
	.byte 1
.else
	.byte 0
.endif

# CHECK-NOT: .byte 0
# CHECK: .byte 1
.ifc foo space, foo space
	.byte 1
.else
	.byte 0
.endif

# CHECK-NOT: .byte 0
# CHECK: .byte 1
.ifc unequal, unEqual
	.byte 0
.else
	.byte 1
.endif

# CHECK-NOT: .byte 0
# CHECK: .byte 1
.ifnc foo, foo
	.byte 0
.else
	.byte 1
.endif

# CHECK-NOT: .byte 0
# CHECK: .byte 1
.ifnc "foo space", "foo space"
	.byte 0
.else
	.byte 1
.endif

# CHECK-NOT: .byte 0
# CHECK: .byte 1
.ifnc foo space, foo space
	.byte 0
.else
	.byte 1
.endif

# CHECK-NOT: .byte 0
# CHECK: .byte 1
.ifnc unequal, unEqual
	.byte 1
.else
	.byte 0
.endif

# CHECK-NOT: .byte 0
# CHECK: .byte 1
.ifnc equal, equal ; .byte 0 ; .else ; .byte 1 ; .endif

