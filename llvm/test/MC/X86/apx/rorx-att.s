# RUN: llvm-mc -triple x86_64 --show-encoding %s | FileCheck %s
# RUN: not llvm-mc -triple i386 -show-encoding %s 2>&1 | FileCheck %s --check-prefix=ERROR

# ERROR-COUNT-4: error:
# ERROR-NOT: error:
# CHECK: rorxl	$123, %r18d, %r22d
# CHECK: encoding: [0x62,0xeb,0x7f,0x08,0xf0,0xf2,0x7b]
         rorxl	$123, %r18d, %r22d

# CHECK: rorxq	$123, %r19, %r23
# CHECK: encoding: [0x62,0xeb,0xff,0x08,0xf0,0xfb,0x7b]
         rorxq	$123, %r19, %r23

# CHECK: rorxl	$123, 291(%r28,%r29,4), %r18d
# CHECK: encoding: [0x62,0x8b,0x7b,0x08,0xf0,0x94,0xac,0x23,0x01,0x00,0x00,0x7b]
         rorxl	$123, 291(%r28,%r29,4), %r18d

# CHECK: rorxq	$123, 291(%r28,%r29,4), %r19
# CHECK: encoding: [0x62,0x8b,0xfb,0x08,0xf0,0x9c,0xac,0x23,0x01,0x00,0x00,0x7b]
         rorxq	$123, 291(%r28,%r29,4), %r19
