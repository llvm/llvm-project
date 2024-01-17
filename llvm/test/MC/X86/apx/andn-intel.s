# RUN: llvm-mc -triple x86_64 -x86-asm-syntax=intel -output-asm-variant=1 --show-encoding %s | FileCheck %s

# CHECK: andn	r26d, r22d, r18d
# CHECK: encoding: [0x62,0x6a,0x4c,0x00,0xf2,0xd2]
         andn	r26d, r22d, r18d

# CHECK: andn	r27, r23, r19
# CHECK: encoding: [0x62,0x6a,0xc4,0x00,0xf2,0xdb]
         andn	r27, r23, r19

# CHECK: andn	r22d, r18d, dword ptr [r28 + 4*r29 + 291]
# CHECK: encoding: [0x62,0x8a,0x68,0x00,0xf2,0xb4,0xac,0x23,0x01,0x00,0x00]
         andn	r22d, r18d, dword ptr [r28 + 4*r29 + 291]

# CHECK: andn	r23, r19, qword ptr [r28 + 4*r29 + 291]
# CHECK: encoding: [0x62,0x8a,0xe0,0x00,0xf2,0xbc,0xac,0x23,0x01,0x00,0x00]
         andn	r23, r19, qword ptr [r28 + 4*r29 + 291]
