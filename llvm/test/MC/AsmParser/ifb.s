# RUN: llvm-mc -triple i386-unknown-unknown %s | FileCheck %s

defined:

# CHECK-NOT: .byte 0
# CHECK: .byte 1
.ifb
	.byte 1
.else
	.byte 0
.endif

# CHECK-NOT: .byte 0
# CHECK: .byte 1
.ifb defined
	.byte 0
.else
	.byte 1
.endif

# CHECK-NOT: .byte 0
# CHECK: .byte 1
.ifb undefined
	.byte 0
.else
	.byte 1
.endif

# CHECK-NOT: .byte 0
# CHECK: .byte 1
.ifb ""
	.byte 0
.else
	.byte 1
.endif

# CHECK-NOT: .byte 0
# CHECK: .byte 1
.ifnb
	.byte 0
.else
	.byte 1
.endif

# CHECK-NOT: .byte 0
# CHECK: .byte 1
.ifnb defined
	.byte 1
.else
	.byte 0
.endif

# CHECK-NOT: .byte 0
# CHECK: .byte 1
.ifnb undefined
	.byte 1
.else
	.byte 0
.endif

# CHECK-NOT: .byte 0
# CHECK: .byte 1
.ifnb ""
	.byte 1
.else
	.byte 0
.endif
