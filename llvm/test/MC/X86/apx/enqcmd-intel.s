# RUN: llvm-mc -triple x86_64 -x86-asm-syntax=intel -output-asm-variant=1 --show-encoding %s | FileCheck %s

## enqcmd

# CHECK: {evex}	enqcmd	ecx, zmmword ptr [eax + 4*ebx + 123]
# CHECK: encoding: [0x67,0x62,0xf4,0x7f,0x08,0xf8,0x4c,0x98,0x7b]
         {evex}	enqcmd	ecx, zmmword ptr [eax + 4*ebx + 123]

# CHECK: {evex}	enqcmd	r9, zmmword ptr [rax + 4*rbx + 123]
# CHECK: encoding: [0x62,0x74,0x7f,0x08,0xf8,0x4c,0x98,0x7b]
         {evex}	enqcmd	r9, zmmword ptr [rax + 4*rbx + 123]

# CHECK: enqcmd	r18d, zmmword ptr [r28d + 4*r29d + 291]
# CHECK: encoding: [0x67,0x62,0x8c,0x7b,0x08,0xf8,0x94,0xac,0x23,0x01,0x00,0x00]
         enqcmd	r18d, zmmword ptr [r28d + 4*r29d + 291]

# CHECK: enqcmd	r19, zmmword ptr [r28 + 4*r29 + 291]
# CHECK: encoding: [0x62,0x8c,0x7b,0x08,0xf8,0x9c,0xac,0x23,0x01,0x00,0x00]
         enqcmd	r19, zmmword ptr [r28 + 4*r29 + 291]

## enqcmds

# CHECK: {evex}	enqcmds	ecx, zmmword ptr [eax + 4*ebx + 123]
# CHECK: encoding: [0x67,0x62,0xf4,0x7e,0x08,0xf8,0x4c,0x98,0x7b]
         {evex}	enqcmds	ecx, zmmword ptr [eax + 4*ebx + 123]

# CHECK: {evex}	enqcmds	r9, zmmword ptr [rax + 4*rbx + 123]
# CHECK: encoding: [0x62,0x74,0x7e,0x08,0xf8,0x4c,0x98,0x7b]
         {evex}	enqcmds	r9, zmmword ptr [rax + 4*rbx + 123]

# CHECK: enqcmds	r18d, zmmword ptr [r28d + 4*r29d + 291]
# CHECK: encoding: [0x67,0x62,0x8c,0x7a,0x08,0xf8,0x94,0xac,0x23,0x01,0x00,0x00]
         enqcmds	r18d, zmmword ptr [r28d + 4*r29d + 291]

# CHECK: enqcmds	r19, zmmword ptr [r28 + 4*r29 + 291]
# CHECK: encoding: [0x62,0x8c,0x7a,0x08,0xf8,0x9c,0xac,0x23,0x01,0x00,0x00]
         enqcmds	r19, zmmword ptr [r28 + 4*r29 + 291]
