# RUN: llvm-mc -triple x86_64 -x86-asm-syntax=intel -output-asm-variant=1 --show-encoding %s | FileCheck %s

# CHECK: {nf}	blsmsk	edx, ecx
# CHECK: encoding: [0x62,0xf2,0x6c,0x0c,0xf3,0xd1]
         {nf}	blsmsk	edx, ecx

# CHECK: {evex}	blsmsk	edx, ecx
# CHECK: encoding: [0x62,0xf2,0x6c,0x08,0xf3,0xd1]
         {evex}	blsmsk	edx, ecx

# CHECK: {nf}	blsmsk	r15, r9
# CHECK: encoding: [0x62,0xd2,0x84,0x0c,0xf3,0xd1]
         {nf}	blsmsk	r15, r9

# CHECK: {evex}	blsmsk	r15, r9
# CHECK: encoding: [0x62,0xd2,0x84,0x08,0xf3,0xd1]
         {evex}	blsmsk	r15, r9

# CHECK: {nf}	blsmsk	ecx, dword ptr [rax + 4*rbx + 123]
# CHECK: encoding: [0x62,0xf2,0x74,0x0c,0xf3,0x54,0x98,0x7b]
         {nf}	blsmsk	ecx, dword ptr [rax + 4*rbx + 123]

# CHECK: {evex}	blsmsk	ecx, dword ptr [rax + 4*rbx + 123]
# CHECK: encoding: [0x62,0xf2,0x74,0x08,0xf3,0x54,0x98,0x7b]
         {evex}	blsmsk	ecx, dword ptr [rax + 4*rbx + 123]

# CHECK: {nf}	blsmsk	r9, qword ptr [rax + 4*rbx + 123]
# CHECK: encoding: [0x62,0xf2,0xb4,0x0c,0xf3,0x54,0x98,0x7b]
         {nf}	blsmsk	r9, qword ptr [rax + 4*rbx + 123]

# CHECK: {evex}	blsmsk	r9, qword ptr [rax + 4*rbx + 123]
# CHECK: encoding: [0x62,0xf2,0xb4,0x08,0xf3,0x54,0x98,0x7b]
         {evex}	blsmsk	r9, qword ptr [rax + 4*rbx + 123]

# CHECK: blsmsk	r22d, r18d
# CHECK: encoding: [0x62,0xfa,0x4c,0x00,0xf3,0xd2]
         blsmsk	r22d, r18d

# CHECK: blsmsk	r23, r19
# CHECK: encoding: [0x62,0xfa,0xc4,0x00,0xf3,0xd3]
         blsmsk	r23, r19

# CHECK: blsmsk	r18d, dword ptr [r28 + 4*r29 + 291]
# CHECK: encoding: [0x62,0x9a,0x68,0x00,0xf3,0x94,0xac,0x23,0x01,0x00,0x00]
         blsmsk	r18d, dword ptr [r28 + 4*r29 + 291]

# CHECK: blsmsk	r19, qword ptr [r28 + 4*r29 + 291]
# CHECK: encoding: [0x62,0x9a,0xe0,0x00,0xf3,0x94,0xac,0x23,0x01,0x00,0x00]
         blsmsk	r19, qword ptr [r28 + 4*r29 + 291]
