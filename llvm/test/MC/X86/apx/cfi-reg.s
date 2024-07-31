// RUN: llvm-mc -triple x86_64-pc-linux-gnu %s -o - | FileCheck %s

f:
	.cfi_startproc
        .cfi_offset 130, -120
        .cfi_offset 131, -112
        .cfi_offset 132, -104
        .cfi_offset 133, -96
        .cfi_offset 134, -88
        .cfi_offset 135, -80
        .cfi_offset 136, -72
        .cfi_offset 137, -64
        .cfi_offset 138, -56
        .cfi_offset 139, -48
        .cfi_offset 140, -40
        .cfi_offset 141, -32
        .cfi_offset 142, -24
        .cfi_offset 143, -16
        .cfi_offset 144, -8
        .cfi_offset 145, 0
	.cfi_endproc

// CHECK: f:
// CHECK-NEXT:         .cfi_startproc
// CHECK-NEXT:         .cfi_offset %r16, -120
// CHECK-NEXT:         .cfi_offset %r17, -112
// CHECK-NEXT:         .cfi_offset %r18, -104
// CHECK-NEXT:         .cfi_offset %r19, -96
// CHECK-NEXT:         .cfi_offset %r20, -88
// CHECK-NEXT:         .cfi_offset %r21, -80
// CHECK-NEXT:         .cfi_offset %r22, -72
// CHECK-NEXT:         .cfi_offset %r23, -64
// CHECK-NEXT:         .cfi_offset %r24, -56
// CHECK-NEXT:         .cfi_offset %r25, -48
// CHECK-NEXT:         .cfi_offset %r26, -40
// CHECK-NEXT:         .cfi_offset %r27, -32
// CHECK-NEXT:         .cfi_offset %r28, -24
// CHECK-NEXT:         .cfi_offset %r29, -16
// CHECK-NEXT:         .cfi_offset %r30, -8
// CHECK-NEXT:         .cfi_offset %r31, 0
// CHECK-NEXT:         .cfi_endproc
