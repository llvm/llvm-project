// RUN: llvm-mc -triple i386 %s | FileCheck %s

.ifeqs "alpha", "alpha"
	.byte 1
.else
	.byte 0
.endif

// CHECK-NOT: .byte 0
// CHECK: .byte 1

.ifeqs "alpha", "alpha "
	.byte 0
.else
	.byte 1
.endif

// CHECK-NOT: .byte 0
// CHECK: .byte 1

