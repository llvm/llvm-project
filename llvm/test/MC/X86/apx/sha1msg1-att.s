# RUN: llvm-mc -triple x86_64 --show-encoding %s | FileCheck %s

# CHECK: sha1msg1	%xmm13, %xmm12
# CHECK: encoding: [0x45,0x0f,0x38,0xc9,0xe5]
         sha1msg1	%xmm13, %xmm12

# CHECK: sha1msg1	291(%r28,%r29,4), %xmm12
# CHECK: encoding: [0x62,0x1c,0x78,0x08,0xd9,0xa4,0xac,0x23,0x01,0x00,0x00]
         sha1msg1	291(%r28,%r29,4), %xmm12
