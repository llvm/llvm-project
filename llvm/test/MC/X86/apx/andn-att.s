# RUN: llvm-mc -triple x86_64 --show-encoding %s | FileCheck %s
# RUN: not llvm-mc -triple i386 -show-encoding %s 2>&1 | FileCheck %s --check-prefix=ERROR

# ERROR-COUNT-4: error:
# ERROR-NOT: error:
# CHECK: andnl	%r18d, %r22d, %r26d
# CHECK: encoding: [0x62,0x6a,0x4c,0x00,0xf2,0xd2]
         andnl	%r18d, %r22d, %r26d

# CHECK: andnq	%r19, %r23, %r27
# CHECK: encoding: [0x62,0x6a,0xc4,0x00,0xf2,0xdb]
         andnq	%r19, %r23, %r27

# CHECK: andnl	291(%r28,%r29,4), %r18d, %r22d
# CHECK: encoding: [0x62,0x8a,0x68,0x00,0xf2,0xb4,0xac,0x23,0x01,0x00,0x00]
         andnl	291(%r28,%r29,4), %r18d, %r22d

# CHECK: andnq	291(%r28,%r29,4), %r19, %r23
# CHECK: encoding: [0x62,0x8a,0xe0,0x00,0xf2,0xbc,0xac,0x23,0x01,0x00,0x00]
         andnq	291(%r28,%r29,4), %r19, %r23
