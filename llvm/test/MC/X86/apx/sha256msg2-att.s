# RUN: llvm-mc -triple x86_64 --show-encoding %s | FileCheck %s

# CHECK: sha256msg2	%xmm13, %xmm12
# CHECK: encoding: [0x45,0x0f,0x38,0xcd,0xe5]
         sha256msg2	%xmm13, %xmm12

# CHECK: sha256msg2	291(%r28,%r29,4), %xmm12
# CHECK: encoding: [0x62,0x1c,0x78,0x08,0xdd,0xa4,0xac,0x23,0x01,0x00,0x00]
         sha256msg2	291(%r28,%r29,4), %xmm12
