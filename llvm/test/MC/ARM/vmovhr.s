// RUN: not llvm-mc -triple=thumbv8.2a-none-eabi -mattr=+fp-armv8,+fullfp16 -show-encoding < %s 2>%t \
// RUN:   | FileCheck %s
// RUN:     FileCheck --check-prefix=ERROR < %t %s

# CHECK: vmov.f16 r0, s13 @ encoding: [0x16,0xee,0x90,0x09]
vmov.f16 r0, s13

# CHECK: vmov.f16 s21, r1 @ encoding: [0x0a,0xee,0x90,0x19]
vmov.f16 s21, r1

# CHECK: vmov.f16 s2, sp @ encoding: [0x01,0xee,0x10,0xd9]
vmov.f16 s2, sp

# ERROR: :[[@LINE+2]]:{{[0-9]+}}: error: invalid instruction
# ERROR: operand must be a register in range [r0, r14]
vmov.f16 s3, pc

# CHECK: vmov.f16 sp, s5 @ encoding: [0x12,0xee,0x90,0xd9]
vmov.f16 sp, s5

# ERROR: :[[@LINE+2]]:{{[0-9]+}}: error: invalid instruction
# ERROR: operand must be a register in range [r0, r14]
vmov.f16 pc, s8

