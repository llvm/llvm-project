# RUN: llvm-mc -triple x86_64 -x86-asm-syntax=intel -output-asm-variant=1 --show-encoding %s | FileCheck %s

# CHECK: invept	r19, xmmword ptr [r28 + 4*r29 + 123]
# CHECK: encoding: [0x62,0x8c,0x7a,0x08,0xf0,0x5c,0xac,0x7b]
         invept	r19, xmmword ptr [r28 + 4*r29 + 123]
