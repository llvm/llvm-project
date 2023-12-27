# RUN: llvm-mc -triple x86_64 -x86-asm-syntax=intel -output-asm-variant=1 --show-encoding %s | FileCheck %s

# CHECK: wrssd	dword ptr [r28 + 4*r29 + 291], r18d
# CHECK: encoding: [0x62,0x8c,0x78,0x08,0x66,0x94,0xac,0x23,0x01,0x00,0x00]
         wrssd	dword ptr [r28 + 4*r29 + 291], r18d
