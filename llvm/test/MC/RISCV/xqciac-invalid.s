# Xqciac - Qualcomm uC Load-Store Address Calculation Extension
# RUN: not llvm-mc -triple riscv32 -mattr=+experimental-xqciac < %s 2>&1 \
# RUN:     | FileCheck -check-prefixes=CHECK,CHECK-IMM %s
# RUN: not llvm-mc -triple riscv32 -mattr=-experimental-xqciac < %s 2>&1 \
# RUN:     | FileCheck -check-prefixes=CHECK,CHECK-EXT %s

# CHECK: :[[@LINE+1]]:14: error: invalid operand for instruction
qc.c.muladdi x5, x10, 4

# CHECK: :[[@LINE+1]]:1: error: too few operands for instruction
qc.c.muladdi x15

# CHECK-IMM: :[[@LINE+1]]:24: error: immediate must be an integer in the range [0, 31]
qc.c.muladdi x10, x15, 32

# CHECK-EXT: :[[@LINE+1]]:1: error: instruction requires the following: 'Xqciac' (Qualcomm uC Load-Store Address Calculation Extension)
qc.c.muladdi x10, x15, 20


# CHECK: :[[@LINE+1]]:12: error: invalid operand for instruction
qc.muladdi x0, x10, 1048577

# CHECK: :[[@LINE+1]]:1: error: too few operands for instruction
qc.muladdi x10

# CHECK-IMM: :[[@LINE+1]]:22: error: operand must be a symbol with %lo/%pcrel_lo/%tprel_lo modifier or an integer in the range [-2048, 2047]
qc.muladdi x10, x15, 8589934592

# CHECK-EXT: :[[@LINE+1]]:1: error: instruction requires the following: 'Xqciac' (Qualcomm uC Load-Store Address Calculation Extension)
qc.muladdi x10, x15, 577


# CHECK: :[[@LINE+1]]:11: error: invalid operand for instruction
qc.shladd 0, x10, 1048577

# CHECK: :[[@LINE+1]]:1: error: too few operands for instruction
qc.shladd x10

# CHECK-IMM: :[[@LINE+1]]:26: error: immediate must be an integer in the range [4, 31]
qc.shladd x10, x15, x11, 2

# CHECK-EXT: :[[@LINE+1]]:1: error: instruction requires the following: 'Xqciac' (Qualcomm uC Load-Store Address Calculation Extension)
qc.shladd x10, x15, x11, 5
