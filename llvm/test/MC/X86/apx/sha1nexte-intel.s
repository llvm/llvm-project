# RUN: llvm-mc -triple x86_64 -x86-asm-syntax=intel -output-asm-variant=1 --show-encoding %s | FileCheck %s

# CHECK: sha1nexte	xmm12, xmm13
# CHECK: encoding: [0x45,0x0f,0x38,0xc8,0xe5]
         sha1nexte	xmm12, xmm13

# CHECK: sha1nexte	xmm12, xmmword ptr [r28 + 4*r29 + 291]
# CHECK: encoding: [0x62,0x1c,0x78,0x08,0xd8,0xa4,0xac,0x23,0x01,0x00,0x00]
         sha1nexte	xmm12, xmmword ptr [r28 + 4*r29 + 291]
