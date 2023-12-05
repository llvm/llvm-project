# RUN: llvm-mc -triple x86_64 --show-encoding %s | FileCheck %s
# RUN: not llvm-mc -triple i386 -show-encoding %s 2>&1 | FileCheck %s --check-prefix=ERROR

# ERROR-COUNT-4: error:
# ERROR-NOT: error:
# CHECK: pextl	%r18d, %r22d, %r26d
# CHECK: encoding: [0x62,0x6a,0x4e,0x00,0xf5,0xd2]
         pextl	%r18d, %r22d, %r26d

# CHECK: pextq	%r19, %r23, %r27
# CHECK: encoding: [0x62,0x6a,0xc6,0x00,0xf5,0xdb]
         pextq	%r19, %r23, %r27

# CHECK: pextl	291(%r28,%r29,4), %r18d, %r22d
# CHECK: encoding: [0x62,0x8a,0x6a,0x00,0xf5,0xb4,0xac,0x23,0x01,0x00,0x00]
         pextl	291(%r28,%r29,4), %r18d, %r22d

# CHECK: pextq	291(%r28,%r29,4), %r19, %r23
# CHECK: encoding: [0x62,0x8a,0xe2,0x00,0xf5,0xbc,0xac,0x23,0x01,0x00,0x00]
         pextq	291(%r28,%r29,4), %r19, %r23
