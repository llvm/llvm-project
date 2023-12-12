# RUN: llvm-mc -triple x86_64 -x86-asm-syntax=intel -output-asm-variant=1 --show-encoding %s | FileCheck %s

# CHECK: bzhi	r26d, r22d, r18d
# CHECK: encoding: [0x62,0x6a,0x6c,0x00,0xf5,0xd6]
         bzhi	r26d, r22d, r18d

# CHECK: bzhi	r22d, dword ptr [r28 + 4*r29 + 291], r18d
# CHECK: encoding: [0x62,0x8a,0x68,0x00,0xf5,0xb4,0xac,0x23,0x01,0x00,0x00]
         bzhi	r22d, dword ptr [r28 + 4*r29 + 291], r18d

# CHECK: bzhi	r27, r23, r19
# CHECK: encoding: [0x62,0x6a,0xe4,0x00,0xf5,0xdf]
         bzhi	r27, r23, r19

# CHECK: bzhi	r23, qword ptr [r28 + 4*r29 + 291], r19
# CHECK: encoding: [0x62,0x8a,0xe0,0x00,0xf5,0xbc,0xac,0x23,0x01,0x00,0x00]
         bzhi	r23, qword ptr [r28 + 4*r29 + 291], r19
