# RUN: llvm-mc -triple x86_64 -x86-asm-syntax=intel -output-asm-variant=1 --show-encoding %s | FileCheck %s

# CHECK: movdiri	dword ptr [r28 + 4*r29 + 291], r18d
# CHECK: encoding: [0x62,0x8c,0x78,0x08,0xf9,0x94,0xac,0x23,0x01,0x00,0x00]
         movdiri	dword ptr [r28 + 4*r29 + 291], r18d

# CHECK: movdiri	qword ptr [r28 + 4*r29 + 291], r19
# CHECK: encoding: [0x62,0x8c,0xf8,0x08,0xf9,0x9c,0xac,0x23,0x01,0x00,0x00]
         movdiri	qword ptr [r28 + 4*r29 + 291], r19
