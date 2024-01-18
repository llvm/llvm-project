# RUN: llvm-mc -triple x86_64 --show-encoding %s | FileCheck %s
# RUN: not llvm-mc -triple i386 -show-encoding %s 2>&1 | FileCheck %s --check-prefix=ERROR

# ERROR-COUNT-4: error:
# ERROR-NOT: error:
# CHECK: mulxl	%r18d, %r22d, %r26d
# CHECK: encoding: [0x62,0x6a,0x4f,0x00,0xf6,0xd2]
         mulxl	%r18d, %r22d, %r26d

# CHECK: mulxq	%r19, %r23, %r27
# CHECK: encoding: [0x62,0x6a,0xc7,0x00,0xf6,0xdb]
         mulxq	%r19, %r23, %r27

# CHECK: mulxl	291(%r28,%r29,4), %r18d, %r22d
# CHECK: encoding: [0x62,0x8a,0x6b,0x00,0xf6,0xb4,0xac,0x23,0x01,0x00,0x00]
         mulxl	291(%r28,%r29,4), %r18d, %r22d

# CHECK: mulxq	291(%r28,%r29,4), %r19, %r23
# CHECK: encoding: [0x62,0x8a,0xe3,0x00,0xf6,0xbc,0xac,0x23,0x01,0x00,0x00]
         mulxq	291(%r28,%r29,4), %r19, %r23
