# RUN: llvm-mc -triple x86_64 -x86-asm-syntax=intel -output-asm-variant=1 --show-encoding %s | FileCheck %s

# CHECK: mulx	r26d, r22d, r18d
# CHECK: encoding: [0x62,0x6a,0x4f,0x00,0xf6,0xd2]
         mulx	r26d, r22d, r18d

# CHECK: mulx	r27, r23, r19
# CHECK: encoding: [0x62,0x6a,0xc7,0x00,0xf6,0xdb]
         mulx	r27, r23, r19

# CHECK: mulx	r22d, r18d, dword ptr [r28 + 4*r29 + 291]
# CHECK: encoding: [0x62,0x8a,0x6b,0x00,0xf6,0xb4,0xac,0x23,0x01,0x00,0x00]
         mulx	r22d, r18d, dword ptr [r28 + 4*r29 + 291]

# CHECK: mulx	r23, r19, qword ptr [r28 + 4*r29 + 291]
# CHECK: encoding: [0x62,0x8a,0xe3,0x00,0xf6,0xbc,0xac,0x23,0x01,0x00,0x00]
         mulx	r23, r19, qword ptr [r28 + 4*r29 + 291]
