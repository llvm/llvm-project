# RUN: llvm-mc -triple x86_64 -x86-asm-syntax=intel -output-asm-variant=1 --show-encoding %s | FileCheck %s

# CHECK: {nf}	andn	r10d, edx, ecx
# CHECK: encoding: [0x62,0x72,0x6c,0x0c,0xf2,0xd1]
         {nf}	andn	r10d, edx, ecx

# CHECK: {evex}	andn	r10d, edx, ecx
# CHECK: encoding: [0x62,0x72,0x6c,0x08,0xf2,0xd1]
         {evex}	andn	r10d, edx, ecx

# CHECK: {nf}	andn	r11, r15, r9
# CHECK: encoding: [0x62,0x52,0x84,0x0c,0xf2,0xd9]
         {nf}	andn	r11, r15, r9

# CHECK: {evex}	andn	r11, r15, r9
# CHECK: encoding: [0x62,0x52,0x84,0x08,0xf2,0xd9]
         {evex}	andn	r11, r15, r9

# CHECK: {nf}	andn	edx, ecx, dword ptr [rax + 4*rbx + 123]
# CHECK: encoding: [0x62,0xf2,0x74,0x0c,0xf2,0x54,0x98,0x7b]
         {nf}	andn	edx, ecx, dword ptr [rax + 4*rbx + 123]

# CHECK: {evex}	andn	edx, ecx, dword ptr [rax + 4*rbx + 123]
# CHECK: encoding: [0x62,0xf2,0x74,0x08,0xf2,0x54,0x98,0x7b]
         {evex}	andn	edx, ecx, dword ptr [rax + 4*rbx + 123]

# CHECK: {nf}	andn	r15, r9, qword ptr [rax + 4*rbx + 123]
# CHECK: encoding: [0x62,0x72,0xb4,0x0c,0xf2,0x7c,0x98,0x7b]
         {nf}	andn	r15, r9, qword ptr [rax + 4*rbx + 123]

# CHECK: {evex}	andn	r15, r9, qword ptr [rax + 4*rbx + 123]
# CHECK: encoding: [0x62,0x72,0xb4,0x08,0xf2,0x7c,0x98,0x7b]
         {evex}	andn	r15, r9, qword ptr [rax + 4*rbx + 123]

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
