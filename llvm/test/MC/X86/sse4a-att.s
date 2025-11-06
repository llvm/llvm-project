# RUN: llvm-mc -triple x86_64-unknown-unknown --show-encoding %s | FileCheck %s

extrq  $2, $3, %xmm0
# CHECK: extrq  $2, $3, %xmm0
# CHECK: encoding: [0x66,0x0f,0x78,0xc0,0x03,0x02]

extrq  %xmm1, %xmm0
# CHECK: extrq  %xmm1, %xmm0
# CHECK: encoding: [0x66,0x0f,0x79,0xc1]

insertq $6, $5, %xmm1, %xmm0
# CHECK: insertq $6, $5, %xmm1, %xmm0
# CHECK: encoding: [0xf2,0x0f,0x78,0xc1,0x05,0x06]

insertq %xmm1, %xmm0
# CHECK: insertq %xmm1, %xmm0
# CHECK: encoding: [0xf2,0x0f,0x79,0xc1]

movntsd %xmm0, (%rdi)
# CHECK: movntsd %xmm0, (%rdi)
# CHECK: encoding: [0xf2,0x0f,0x2b,0x07]

movntss %xmm0, (%rdi)
# CHECK: movntss %xmm0, (%rdi)
# CHECK: encoding: [0xf3,0x0f,0x2b,0x07]
