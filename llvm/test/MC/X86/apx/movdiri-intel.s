# RUN: llvm-mc -triple x86_64 -x86-asm-syntax=intel -output-asm-variant=1 --show-encoding %s | FileCheck %s

# CHECK: {evex}	movdiri	dword ptr [eax + 4*ebx + 123], ecx
# CHECK: encoding: [0x67,0x62,0xf4,0x7c,0x08,0xf9,0x4c,0x98,0x7b]
         {evex}	movdiri	dword ptr [eax + 4*ebx + 123], ecx

# CHECK: {evex}	movdiri	qword ptr [rax + 4*rbx + 123], r9
# CHECK: encoding: [0x62,0x74,0xfc,0x08,0xf9,0x4c,0x98,0x7b]
         {evex}	movdiri	qword ptr [rax + 4*rbx + 123], r9

# CHECK: movdiri	dword ptr [r28 + 4*r29 + 291], r18d
# CHECK: encoding: [0x62,0x8c,0x78,0x08,0xf9,0x94,0xac,0x23,0x01,0x00,0x00]
         movdiri	dword ptr [r28 + 4*r29 + 291], r18d

# CHECK: movdiri	qword ptr [r28 + 4*r29 + 291], r19
# CHECK: encoding: [0x62,0x8c,0xf8,0x08,0xf9,0x9c,0xac,0x23,0x01,0x00,0x00]
         movdiri	qword ptr [r28 + 4*r29 + 291], r19
