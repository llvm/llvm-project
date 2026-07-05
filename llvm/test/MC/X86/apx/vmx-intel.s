# RUN: llvm-mc -triple x86_64 -x86-asm-syntax=intel -output-asm-variant=1 --show-encoding %s | FileCheck %s

## invept

# CHECK: invept	r19, xmmword ptr [r28 + 4*r29 + 291]
# CHECK: encoding: [0x62,0x8c,0x7a,0x08,0xf0,0x9c,0xac,0x23,0x01,0x00,0x00]
         invept	r19, xmmword ptr [r28 + 4*r29 + 291]

# CHECK: {evex}	invept	r9, xmmword ptr [rax + 4*rbx + 123]
# CHECK: encoding: [0x62,0x74,0x7e,0x08,0xf0,0x4c,0x98,0x7b]
         {evex}	invept	r9, xmmword ptr [rax + 4*rbx + 123]

## invvpid

# CHECK: invvpid	r19, xmmword ptr [r28 + 4*r29 + 291]
# CHECK: encoding: [0x62,0x8c,0x7a,0x08,0xf1,0x9c,0xac,0x23,0x01,0x00,0x00]
         invvpid	r19, xmmword ptr [r28 + 4*r29 + 291]

# CHECK: {evex}	invvpid	r9, xmmword ptr [rax + 4*rbx + 123]
# CHECK: encoding: [0x62,0x74,0x7e,0x08,0xf1,0x4c,0x98,0x7b]
         {evex}	invvpid	r9, xmmword ptr [rax + 4*rbx + 123]
