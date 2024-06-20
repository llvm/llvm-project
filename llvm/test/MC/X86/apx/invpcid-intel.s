# RUN: llvm-mc -triple x86_64 -x86-asm-syntax=intel -output-asm-variant=1 --show-encoding %s | FileCheck %s

# CHECK: {evex}	invpcid	r9, xmmword ptr [rax + 4*rbx + 123]
# CHECK: encoding: [0x62,0x74,0x7e,0x08,0xf2,0x4c,0x98,0x7b]
         {evex}	invpcid	r9, xmmword ptr [rax + 4*rbx + 123]

# CHECK: invpcid	r19, xmmword ptr [r28 + 4*r29 + 291]
# CHECK: encoding: [0x62,0x8c,0x7a,0x08,0xf2,0x9c,0xac,0x23,0x01,0x00,0x00]
         invpcid	r19, xmmword ptr [r28 + 4*r29 + 291]
