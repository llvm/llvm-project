# RUN: llvm-mc -triple x86_64 -x86-asm-syntax=intel -output-asm-variant=1 --show-encoding %s | FileCheck %s

# CHECK: pdep	r26d, r22d, r18d
# CHECK: encoding: [0x62,0x6a,0x4f,0x00,0xf5,0xd2]
         pdep	r26d, r22d, r18d

# CHECK: pdep	r27, r23, r19
# CHECK: encoding: [0x62,0x6a,0xc7,0x00,0xf5,0xdb]
         pdep	r27, r23, r19

# CHECK: pdep	r22d, r18d, dword ptr [r28 + 4*r29 + 291]
# CHECK: encoding: [0x62,0x8a,0x6b,0x00,0xf5,0xb4,0xac,0x23,0x01,0x00,0x00]
         pdep	r22d, r18d, dword ptr [r28 + 4*r29 + 291]

# CHECK: pdep	r23, r19, qword ptr [r28 + 4*r29 + 291]
# CHECK: encoding: [0x62,0x8a,0xe3,0x00,0xf5,0xbc,0xac,0x23,0x01,0x00,0x00]
         pdep	r23, r19, qword ptr [r28 + 4*r29 + 291]
