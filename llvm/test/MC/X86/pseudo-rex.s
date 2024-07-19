# RUN: llvm-mc -triple x86_64 -show-encoding %s | FileCheck %s

# CHECK: addl    %ebx, %ecx
# CHECK: encoding: [0x40,0x01,0xd9]
{rex} addl %ebx, %ecx

# CHECK: popcntl %edi, %esi
# CHECK: encoding: [0xf3,0x40,0x0f,0xb8,0xf7]
{rex} popcnt %edi,%esi
