# RUN: llvm-mc -triple x86_64 --show-encoding %s | FileCheck %s
# RUN: not llvm-mc -triple i386 -show-encoding %s 2>&1 | FileCheck %s --check-prefix=ERROR

# ERROR-COUNT-4: error:
# ERROR-NOT: error:
# CHECK: bextrl	%r18d, %r22d, %r26d
# CHECK: encoding: [0x62,0x6a,0x6c,0x00,0xf7,0xd6]
         bextrl	%r18d, %r22d, %r26d

# CHECK: bextrl	%r18d, 291(%r28,%r29,4), %r22d
# CHECK: encoding: [0x62,0x8a,0x68,0x00,0xf7,0xb4,0xac,0x23,0x01,0x00,0x00]
         bextrl	%r18d, 291(%r28,%r29,4), %r22d

# CHECK: bextrq	%r19, %r23, %r27
# CHECK: encoding: [0x62,0x6a,0xe4,0x00,0xf7,0xdf]
         bextrq	%r19, %r23, %r27

# CHECK: bextrq	%r19, 291(%r28,%r29,4), %r23
# CHECK: encoding: [0x62,0x8a,0xe0,0x00,0xf7,0xbc,0xac,0x23,0x01,0x00,0x00]
         bextrq	%r19, 291(%r28,%r29,4), %r23
