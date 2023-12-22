# RUN: llvm-mc -triple x86_64 --show-encoding %s | FileCheck %s

# CHECK: tilestored	%tmm6, 291(%r28,%r29,4)
# CHECK: encoding: [0x62,0x9a,0x7a,0x08,0x4b,0xb4,0xac,0x23,0x01,0x00,0x00]
         tilestored	%tmm6, 291(%r28,%r29,4)
