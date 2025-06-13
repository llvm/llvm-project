# Xqcibi - Qualcomm uC Branch Immediate Extension
# RUN: not llvm-mc -triple riscv32 -mattr=+experimental-xqcibi < %s 2>&1 \
# RUN:     | FileCheck -check-prefixes=CHECK,CHECK-PLUS %s
# RUN: not llvm-mc -triple riscv32 -mattr=-experimental-xqcibi < %s 2>&1 \
# RUN:     | FileCheck -check-prefixes=CHECK,CHECK-MINUS %s

# CHECK-PLUS: :[[@LINE+2]]:9: error: register must be a GPR excluding zero (x0)
# CHECK-MINUS: :[[@LINE+1]]:9: error: invalid operand for instruction
qc.beqi x0, 12, 346

# CHECK: :[[@LINE+1]]:1: error: too few operands for instruction
qc.beqi x8, 12

# CHECK-PLUS: :[[@LINE+1]]:13: error: immediate must be non-zero in the range [-16, 15]
qc.beqi x8, 22, 346

# CHECK-PLUS: :[[@LINE+1]]:17: error: immediate must be a multiple of 2 bytes in the range [-4096, 4094]
qc.beqi x8, 12, 1211

# CHECK-MINUS: :[[@LINE+1]]:1: error: instruction requires the following: 'Xqcibi' (Qualcomm uC Branch Immediate Extension)
qc.beqi x8, 12, 346


# CHECK-PLUS: :[[@LINE+2]]:9: error: register must be a GPR excluding zero (x0)
# CHECK-MINUS: :[[@LINE+1]]:9: error: invalid operand for instruction
qc.bnei x0, 15, 4094

# CHECK: :[[@LINE+1]]:1: error: too few operands for instruction
qc.bnei x4, 15

# CHECK-PLUS: :[[@LINE+1]]:13: error: immediate must be non-zero in the range [-16, 15]
qc.bnei x4, -45, 4094

# CHECK-PLUS: :[[@LINE+1]]:17: error: immediate must be a multiple of 2 bytes in the range [-4096, 4094]
qc.bnei x4, 15, 5000

# CHECK-MINUS: :[[@LINE+1]]:1: error: instruction requires the following: 'Xqcibi' (Qualcomm uC Branch Immediate Extension)
qc.bnei x4, 15, 4094


# CHECK-PLUS: :[[@LINE+2]]:9: error: register must be a GPR excluding zero (x0)
# CHECK-MINUS: :[[@LINE+1]]:9: error: invalid operand for instruction
qc.bgei x0, 1, -4096

# CHECK: :[[@LINE+1]]:1: error: too few operands for instruction
qc.bgei x10, 1

# CHECK-PLUS: :[[@LINE+1]]:14: error: immediate must be non-zero in the range [-16, 15]
qc.bgei x10, 21, -4096

# CHECK-PLUS: :[[@LINE+1]]:17: error: immediate must be a multiple of 2 bytes in the range [-4096, 4094]
qc.bgei x10, 1, -4098

# CHECK-MINUS: :[[@LINE+1]]:1: error: instruction requires the following: 'Xqcibi' (Qualcomm uC Branch Immediate Extension)
qc.bgei x10, 1, -4096


# CHECK-PLUS: :[[@LINE+2]]:9: error: register must be a GPR excluding zero (x0)
# CHECK-MINUS: :[[@LINE+1]]:9: error: invalid operand for instruction
qc.blti x0, 6, 2000

# CHECK: :[[@LINE+1]]:1: error: too few operands for instruction
qc.blti x1, 6

# CHECK-PLUS: :[[@LINE+1]]:13: error: immediate must be non-zero in the range [-16, 15]
qc.blti x1, 56, 2000

# CHECK-PLUS: :[[@LINE+1]]:16: error: immediate must be a multiple of 2 bytes in the range [-4096, 4094]
qc.blti x1, 6, 12000

# CHECK-MINUS: :[[@LINE+1]]:1: error: instruction requires the following: 'Xqcibi' (Qualcomm uC Branch Immediate Extension)
qc.blti x1, 6, 2000


# CHECK-PLUS: :[[@LINE+2]]:10: error: register must be a GPR excluding zero (x0)
# CHECK-MINUS: :[[@LINE+1]]:10: error: invalid operand for instruction
qc.bgeui x0, 11, 128

# CHECK: :[[@LINE+1]]:1: error: too few operands for instruction
qc.bgeui x12, 11

# CHECK-PLUS: :[[@LINE+1]]:15: error: immediate must be an integer in the range [1, 31]
qc.bgeui x12, 41, 128

# CHECK-PLUS: :[[@LINE+1]]:19: error: immediate must be a multiple of 2 bytes in the range [-4096, 4094]
qc.bgeui x12, 11, 11128

# CHECK-MINUS: :[[@LINE+1]]:1: error: instruction requires the following: 'Xqcibi' (Qualcomm uC Branch Immediate Extension)
qc.bgeui x12, 11, 128


# CHECK-PLUS: :[[@LINE+2]]:10: error: register must be a GPR excluding zero (x0)
# CHECK-MINUS: :[[@LINE+1]]:10: error: invalid operand for instruction
qc.bltui x0, 7, 666

# CHECK: :[[@LINE+1]]:1: error: too few operands for instruction
qc.bltui x2, 7

# CHECK-PLUS: :[[@LINE+1]]:14: error: immediate must be an integer in the range [1, 31]
qc.bltui x2, -7, 666

# CHECK-PLUS: :[[@LINE+1]]:17: error: immediate must be a multiple of 2 bytes in the range [-4096, 4094]
qc.bltui x2, 7, -6666

# CHECK-MINUS: :[[@LINE+1]]:1: error: instruction requires the following: 'Xqcibi' (Qualcomm uC Branch Immediate Extension)
qc.bltui x2, 7, 666


# CHECK-PLUS: :[[@LINE+2]]:11: error: register must be a GPR excluding zero (x0)
# CHECK-MINUS: :[[@LINE+1]]:11: error: invalid operand for instruction
qc.e.beqi x0, 1, 2

# CHECK: :[[@LINE+1]]:1: error: too few operands for instruction
qc.e.beqi x1, 1

# CHECK-PLUS: :[[@LINE+1]]:15: error: immediate must be non-zero in the range [-32768, 32767]
qc.e.beqi x1, 32768, 2

# CHECK-PLUS: :[[@LINE+1]]:18: error: immediate must be a multiple of 2 bytes in the range [-4096, 4094]
qc.e.beqi x1, 1, 21

# CHECK-MINUS: :[[@LINE+1]]:1: error: instruction requires the following: 'Xqcibi' (Qualcomm uC Branch Immediate Extension)
qc.e.beqi x1, 1, 2


# CHECK-PLUS: :[[@LINE+2]]:11: error: register must be a GPR excluding zero (x0)
# CHECK-MINUS: :[[@LINE+1]]:11: error: invalid operand for instruction
qc.e.bnei x0, 115, 4094

# CHECK: :[[@LINE+1]]:1: error: too few operands for instruction
qc.e.bnei x4, 115

# CHECK-PLUS: :[[@LINE+1]]:15: error: immediate must be non-zero in the range [-32768, 32767]
qc.e.bnei x4, -33115, 4094

# CHECK-PLUS: :[[@LINE+1]]:20: error: immediate must be a multiple of 2 bytes in the range [-4096, 4094]
qc.e.bnei x4, 115, 211

# CHECK-MINUS: :[[@LINE+1]]:1: error: instruction requires the following: 'Xqcibi' (Qualcomm uC Branch Immediate Extension)
qc.e.bnei x4, 115, 4094


# CHECK-PLUS: :[[@LINE+2]]:11: error: register must be a GPR excluding zero (x0)
# CHECK-MINUS: :[[@LINE+1]]:11: error: invalid operand for instruction
qc.e.bgei x0, -32768, -4096

# CHECK: :[[@LINE+1]]:1: error: too few operands for instruction
qc.e.bgei x10, -32768

# CHECK-PLUS: :[[@LINE+1]]:16: error: immediate must be non-zero in the range [-32768, 32767]
qc.e.bgei x10, -32769, -4096

# CHECK-PLUS: :[[@LINE+1]]:24: error: immediate must be a multiple of 2 bytes in the range [-4096, 4094]
qc.e.bgei x10, -32768, -4097

# CHECK-MINUS: :[[@LINE+1]]:1: error: instruction requires the following: 'Xqcibi' (Qualcomm uC Branch Immediate Extension)
qc.e.bgei x10, -32768, -4096


# CHECK-PLUS: :[[@LINE+2]]:11: error: register must be a GPR excluding zero (x0)
# CHECK-MINUS: :[[@LINE+1]]:11: error: invalid operand for instruction
qc.e.blti x0, 32767, 2000

# CHECK: :[[@LINE+1]]:1: error: too few operands for instruction
qc.e.blti x1, 32767

# CHECK-PLUS: :[[@LINE+1]]:15: error: immediate must be non-zero in the range [-32768, 32767]
qc.e.blti x1, 42767, 2000

# CHECK-PLUS: :[[@LINE+1]]:22: error: immediate must be a multiple of 2 bytes in the range [-4096, 4094]
qc.e.blti x1, 32767, 2001

# CHECK-MINUS: :[[@LINE+1]]:1: error: instruction requires the following: 'Xqcibi' (Qualcomm uC Branch Immediate Extension)
qc.e.blti x1, 32767, 2000


# CHECK-PLUS: :[[@LINE+2]]:12: error: register must be a GPR excluding zero (x0)
# CHECK-MINUS: :[[@LINE+1]]:12: error: invalid operand for instruction
qc.e.bgeui x0, 711, 128

# CHECK: :[[@LINE+1]]:1: error: too few operands for instruction
qc.e.bgeui x12, 711

# CHECK-PLUS: :[[@LINE+1]]:17: error: immediate must be an integer in the range [1, 65535]
qc.e.bgeui x12, 0, 128

# CHECK-PLUS: :[[@LINE+1]]:22: error: immediate must be a multiple of 2 bytes in the range [-4096, 4094]
qc.e.bgeui x12, 711, 129

# CHECK-MINUS: :[[@LINE+1]]:1: error: instruction requires the following: 'Xqcibi' (Qualcomm uC Branch Immediate Extension)
qc.e.bgeui x12, 711, 128


# CHECK-PLUS: :[[@LINE+2]]:12: error: register must be a GPR excluding zero (x0)
# CHECK-MINUS: :[[@LINE+1]]:12: error: invalid operand for instruction
qc.e.bltui x0, 7, 666

# CHECK: :[[@LINE+1]]:1: error: too few operands for instruction
qc.e.bltui x2, 7

# CHECK-PLUS: :[[@LINE+1]]:16: error: immediate must be an integer in the range [1, 65535]
qc.e.bltui x2, -7, 666

# CHECK-PLUS: :[[@LINE+1]]:19: error: immediate must be a multiple of 2 bytes in the range [-4096, 4094]
qc.e.bltui x2, 7, 667

# CHECK-MINUS: :[[@LINE+1]]:1: error: instruction requires the following: 'Xqcibi' (Qualcomm uC Branch Immediate Extension)
qc.e.bltui x2, 7, 666
