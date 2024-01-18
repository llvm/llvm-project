# RUN: llvm-mc -triple x86_64 -x86-asm-syntax=intel -output-asm-variant=1 --show-encoding %s | FileCheck %s

# CHECK: pext	r26d, r22d, r18d
# CHECK: encoding: [0x62,0x6a,0x4e,0x00,0xf5,0xd2]
         pext	r26d, r22d, r18d

# CHECK: pext	r27, r23, r19
# CHECK: encoding: [0x62,0x6a,0xc6,0x00,0xf5,0xdb]
         pext	r27, r23, r19

# CHECK: pext	r22d, r18d, dword ptr [r28 + 4*r29 + 291]
# CHECK: encoding: [0x62,0x8a,0x6a,0x00,0xf5,0xb4,0xac,0x23,0x01,0x00,0x00]
         pext	r22d, r18d, dword ptr [r28 + 4*r29 + 291]

# CHECK: pext	r23, r19, qword ptr [r28 + 4*r29 + 291]
# CHECK: encoding: [0x62,0x8a,0xe2,0x00,0xf5,0xbc,0xac,0x23,0x01,0x00,0x00]
         pext	r23, r19, qword ptr [r28 + 4*r29 + 291]
