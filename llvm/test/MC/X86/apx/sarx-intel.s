# RUN: llvm-mc -triple x86_64 -x86-asm-syntax=intel -output-asm-variant=1 --show-encoding %s | FileCheck %s

# CHECK: sarx	r26d, r22d, r18d
# CHECK: encoding: [0x62,0x6a,0x6e,0x00,0xf7,0xd6]
         sarx	r26d, r22d, r18d

# CHECK: sarx	r22d, dword ptr [r28 + 4*r29 + 291], r18d
# CHECK: encoding: [0x62,0x8a,0x6a,0x00,0xf7,0xb4,0xac,0x23,0x01,0x00,0x00]
         sarx	r22d, dword ptr [r28 + 4*r29 + 291], r18d

# CHECK: sarx	r27, r23, r19
# CHECK: encoding: [0x62,0x6a,0xe6,0x00,0xf7,0xdf]
         sarx	r27, r23, r19

# CHECK: sarx	r23, qword ptr [r28 + 4*r29 + 291], r19
# CHECK: encoding: [0x62,0x8a,0xe2,0x00,0xf7,0xbc,0xac,0x23,0x01,0x00,0x00]
         sarx	r23, qword ptr [r28 + 4*r29 + 291], r19
