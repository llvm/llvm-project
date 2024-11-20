# RUN: llvm-mc -triple x86_64-unknown-unknown %s | FileCheck %s

.intel_syntax

# CHECK:  vinsertps {{.*}} # xmm2 = xmm2[0,1,2],mem[0]

vinsertps xmm2,xmm2,dword ptr [r14+rdi*8+0x4C],0x0B0
