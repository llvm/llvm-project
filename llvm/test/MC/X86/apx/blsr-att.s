# RUN: llvm-mc -triple x86_64 --show-encoding %s | FileCheck %s
# RUN: not llvm-mc -triple i386 -show-encoding %s 2>&1 | FileCheck %s --check-prefix=ERROR

# ERROR-COUNT-4: error:
# ERROR-NOT: error:
# CHECK: blsrl	%r18d, %r22d
# CHECK: encoding: [0x62,0xfa,0x4c,0x00,0xf3,0xca]
         blsrl	%r18d, %r22d

# CHECK: blsrq	%r19, %r23
# CHECK: encoding: [0x62,0xfa,0xc4,0x00,0xf3,0xcb]
         blsrq	%r19, %r23

# CHECK: blsrl	291(%r28,%r29,4), %r18d
# CHECK: encoding: [0x62,0x9a,0x68,0x00,0xf3,0x8c,0xac,0x23,0x01,0x00,0x00]
         blsrl	291(%r28,%r29,4), %r18d

# CHECK: blsrq	291(%r28,%r29,4), %r19
# CHECK: encoding: [0x62,0x9a,0xe0,0x00,0xf3,0x8c,0xac,0x23,0x01,0x00,0x00]
         blsrq	291(%r28,%r29,4), %r19
