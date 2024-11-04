# RUN: llvm-mc -triple x86_64 -show-encoding %s | FileCheck %s

# CHECK: addl    %ebx, %ecx
# CHECK: encoding: [0xd5,0x00,0x01,0xd9]
{rex2} addl %ebx, %ecx

# CHECK: popcntl %edi, %esi
# CHECK: encoding: [0xf3,0xd5,0x80,0xb8,0xf7]
{rex2} popcnt %edi,%esi
