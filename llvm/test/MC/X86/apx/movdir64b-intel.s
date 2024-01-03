# RUN: llvm-mc -triple x86_64 -x86-asm-syntax=intel -output-asm-variant=1 --show-encoding %s | FileCheck %s

# CHECK: movdir64b	r18d, zmmword ptr [r28d + 4*r29d + 291]
# CHECK: encoding: [0x67,0x62,0x8c,0x79,0x08,0xf8,0x94,0xac,0x23,0x01,0x00,0x00]
         movdir64b	r18d, zmmword ptr [r28d + 4*r29d + 291]

# CHECK: movdir64b	r19, zmmword ptr [r28 + 4*r29 + 291]
# CHECK: encoding: [0x62,0x8c,0x79,0x08,0xf8,0x9c,0xac,0x23,0x01,0x00,0x00]
         movdir64b	r19, zmmword ptr [r28 + 4*r29 + 291]
