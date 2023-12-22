# RUN: llvm-mc -triple x86_64 --show-encoding %s | FileCheck %s
# RUN: not llvm-mc -triple i386 -show-encoding %s 2>&1 | FileCheck %s --check-prefix=ERROR

# ERROR-COUNT-1: error:
# ERROR-NOT: error:
# CHECK: wrssq	%r19, 291(%r28,%r29,4)
# CHECK: encoding: [0x62,0x8c,0xf8,0x08,0x66,0x9c,0xac,0x23,0x01,0x00,0x00]
         wrssq	%r19, 291(%r28,%r29,4)
