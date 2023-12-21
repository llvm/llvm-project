# RUN: llvm-mc -triple x86_64 -x86-asm-syntax=intel -output-asm-variant=1 --show-encoding %s | FileCheck %s

# CHECK: shrx	r26d, r22d, r18d
# CHECK: encoding: [0x62,0x6a,0x6f,0x00,0xf7,0xd6]
         shrx	r26d, r22d, r18d

# CHECK: shrx	r22d, dword ptr [r28 + 4*r29 + 291], r18d
# CHECK: encoding: [0x62,0x8a,0x6b,0x00,0xf7,0xb4,0xac,0x23,0x01,0x00,0x00]
         shrx	r22d, dword ptr [r28 + 4*r29 + 291], r18d

# CHECK: shrx	r27, r23, r19
# CHECK: encoding: [0x62,0x6a,0xe7,0x00,0xf7,0xdf]
         shrx	r27, r23, r19

# CHECK: shrx	r23, qword ptr [r28 + 4*r29 + 291], r19
# CHECK: encoding: [0x62,0x8a,0xe3,0x00,0xf7,0xbc,0xac,0x23,0x01,0x00,0x00]
         shrx	r23, qword ptr [r28 + 4*r29 + 291], r19
