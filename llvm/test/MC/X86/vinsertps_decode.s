# RUN: llvm-mc -triple x86_64-unknown-unknown %s | FileCheck %s

.intel_syntax

# CHECK: insertps  $176, (%rax), %xmm2        # xmm2 = xmm2[0,1,2],mem[0]
# CHECK: vinsertps $176, (%rax), %xmm2, %xmm2 # xmm2 = xmm2[0,1,2],mem[0]
# CHECK: vinsertps $176, (%rax), %xmm29, %xmm0 # xmm0 = xmm29[0,1,2],mem[0]

insertps xmm2, dword ptr [rax], 0x0B0
vinsertps xmm2,xmm2,dword ptr [rax],0x0B0
vinsertps xmm0,xmm29,dword ptr [rax],0x0B0
