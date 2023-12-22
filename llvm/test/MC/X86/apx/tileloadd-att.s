# RUN: llvm-mc -triple x86_64 --show-encoding %s | FileCheck %s

# CHECK: tileloadd	291(%r28,%r29,4), %tmm6
# CHECK: encoding: [0x62,0x9a,0x7b,0x08,0x4b,0xb4,0xac,0x23,0x01,0x00,0x00]
         tileloadd	291(%r28,%r29,4), %tmm6
