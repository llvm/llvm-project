# RUN: not llvm-mc -triple riscv32 -mattr=+xwchc < %s 2>&1 | FileCheck %s

qk.c.lbu x8, 32(x8) # CHECK: :[[@LINE]]:{{[0-9]+}}: error: immediate must be an integer in the range [0, 31]
qk.c.sb x8, 32(x8) # CHECK: :[[@LINE]]:{{[0-9]+}}: error: immediate must be an integer in the range [0, 31]

qk.c.lhu x8, 1(x8) # CHECK: :[[@LINE]]:{{[0-9]+}}: error: immediate must be a multiple of 2 bytes in the range [0, 62]
qk.c.sh x8, 1(x8) # CHECK: :[[@LINE]]:{{[0-9]+}}: error: immediate must be a multiple of 2 bytes in the range [0, 62]
qk.c.lhu x8, 64(x8) # CHECK: :[[@LINE]]:{{[0-9]+}}: error: immediate must be a multiple of 2 bytes in the range [0, 62]
qk.c.sh x8, 64(x8) # CHECK: :[[@LINE]]:{{[0-9]+}}: error: immediate must be a multiple of 2 bytes in the range [0, 62]

qk.c.lbusp x8, 0(x8) # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
qk.c.sbsp x8, 0(x8) # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
qk.c.lbusp x8, 32(sp) # CHECK: :[[@LINE]]:{{[0-9]+}}: error: immediate must be an integer in the range [0, 15]
qk.c.sbsp x8, 32(sp) # CHECK: :[[@LINE]]:{{[0-9]+}}: error: immediate must be an integer in the range [0, 15]

qk.c.lhusp x8, 0(x8) # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
qk.c.shsp x8, 0(x8) # CHECK: :[[@LINE]]:{{[0-9]+}}: error: invalid operand for instruction
qk.c.lhusp x8, 1(sp) # CHECK: :[[@LINE]]:{{[0-9]+}}: error: immediate must be a multiple of 2 bytes in the range [0, 30]
qk.c.shsp x8, 1(sp) # CHECK: :[[@LINE]]:{{[0-9]+}}: error: immediate must be a multiple of 2 bytes in the range [0, 30]
qk.c.lhusp x8, 32(sp) # CHECK: :[[@LINE]]:{{[0-9]+}}: error: immediate must be a multiple of 2 bytes in the range [0, 30]
qk.c.shsp x8, 32(sp) # CHECK: :[[@LINE]]:{{[0-9]+}}: error: immediate must be a multiple of 2 bytes in the range [0, 30]
