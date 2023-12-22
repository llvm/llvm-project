# RUN: llvm-mc -triple x86_64 -x86-asm-syntax=intel -output-asm-variant=1 --show-encoding %s | FileCheck %s

# CHECK: tilestored	[r28 + 4*r29 + 291], tmm6
# CHECK: encoding: [0x62,0x9a,0x7a,0x08,0x4b,0xb4,0xac,0x23,0x01,0x00,0x00]
         tilestored	[r28 + 4*r29 + 291], tmm6
