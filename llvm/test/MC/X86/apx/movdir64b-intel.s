# RUN: llvm-mc -triple x86_64 -x86-asm-syntax=intel -output-asm-variant=1 --show-encoding %s | FileCheck %s

# CHECK: {evex}	movdir64b	ecx, zmmword ptr [eax + 4*ebx + 123]
# CHECK: encoding: [0x67,0x62,0xf4,0x7d,0x08,0xf8,0x4c,0x98,0x7b]
         {evex}	movdir64b	ecx, zmmword ptr [eax + 4*ebx + 123]

# CHECK: {evex}	movdir64b	r9, zmmword ptr [rax + 4*rbx + 123]
# CHECK: encoding: [0x62,0x74,0x7d,0x08,0xf8,0x4c,0x98,0x7b]
         {evex}	movdir64b	r9, zmmword ptr [rax + 4*rbx + 123]

# CHECK: movdir64b	r18d, zmmword ptr [r28d + 4*r29d + 291]
# CHECK: encoding: [0x67,0x62,0x8c,0x79,0x08,0xf8,0x94,0xac,0x23,0x01,0x00,0x00]
         movdir64b	r18d, zmmword ptr [r28d + 4*r29d + 291]

# CHECK: movdir64b	r19, zmmword ptr [r28 + 4*r29 + 291]
# CHECK: encoding: [0x62,0x8c,0x79,0x08,0xf8,0x9c,0xac,0x23,0x01,0x00,0x00]
         movdir64b	r19, zmmword ptr [r28 + 4*r29 + 291]
