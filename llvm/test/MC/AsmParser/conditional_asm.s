# RUN: llvm-mc -triple i386-unknown-unknown %s -I  %p | FileCheck %s

# CHECK: .byte 2
.if 1+2
    .if 1-1
        .byte 1
    .elseif 2+2
        .byte 1+1
    .else
        .byte 0
    .endif
.endif

# CHECK: .byte 0
# CHECK-NOT: .byte 1
.ifeq 32 - 32
        .byte 0
.else
        .byte 1
.endif

# CHECK: .byte 0
# CHECK: .byte 1
# CHECK-NOT: .byte 2
.ifge 32 - 31
        .byte 0
.endif
.ifge 32 - 32
        .byte 1
.endif
.ifge 32 - 33
        .byte 2
.endif

# CHECK: .byte 0
# CHECK-NOT: .byte 1
# CHECK-NOT: .byte 2
.ifgt 32 - 31
        .byte 0
.endif
.ifgt 32 - 32
        .byte 1
.endif
.ifgt 32 - 33
        .byte 2
.endif

# CHECK-NOT: .byte 0
# CHECK: .byte 1
# CHECK: .byte 2
.ifle 32 - 31
        .byte 0
.endif
.ifle 32 - 32
        .byte 1
.endif
.ifle 32 - 33
        .byte 2
.endif

# CHECK-NOT: .byte 0
# CHECK-NOT: .byte 1
# CHECK: .byte 2
.iflt 32 - 31
        .byte 0
.endif
.iflt 32 - 32
        .byte 1
.endif
.iflt 32 - 33
        .byte 2
.endif

# CHECK: .byte 1
# CHECK-NOT: .byte 0
.ifne 32 - 32
	.byte 0
.else
	.byte 1
.endif

