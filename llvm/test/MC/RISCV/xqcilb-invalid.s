# Xqcilb - Qualcomm uC Long Branch Extension
# RUN: not llvm-mc -triple riscv32 -mattr=+xqcilb < %s 2>&1 \
# RUN:     | FileCheck --match-full-lines -check-prefixes=CHECK,CHECK-PLUS %s
# RUN: not llvm-mc -triple riscv32 -mattr=-xqcilb < %s 2>&1 \
# RUN:     | FileCheck --match-full-lines -check-prefixes=CHECK,CHECK-MINUS %s

# CHECK-PLUS: <stdin>:[[@LINE+2]]:1: error: too few operands for instruction
# CHECK-MINUS: <stdin>:[[@LINE+1]]:1: error: invalid instruction
qc.e.j

# CHECK-PLUS: <stdin>:[[@LINE+1]]:9: error: operand must be a multiple of 2 bytes in the range [-2147483648, 2147483646]
qc.e.j  -2147483649

# CHECK-MINUS: <stdin>:[[@LINE+1]]:1: error: instruction requires the following: 'Xqcilb' (Qualcomm uC Long Branch Extension)
qc.e.j  -2147483648

# CHECK-MINUS: <stdin>:[[@LINE+1]]:1: error: instruction requires the following: 'Xqcilb' (Qualcomm uC Long Branch Extension)
qc.e.j  foo

# CHECK-PLUS: <stdin>:[[@LINE+2]]:1: error: too few operands for instruction
# CHECK-MINUS: <stdin>:[[@LINE+1]]:1: error: invalid instruction
qc.e.jal

# CHECK-PLUS: <stdin>:[[@LINE+1]]:10: error: operand must be a multiple of 2 bytes in the range [-2147483648, 2147483646]
qc.e.jal 2147483649

# CHECK-MINUS: <stdin>:[[@LINE+1]]:1: error: instruction requires the following: 'Xqcilb' (Qualcomm uC Long Branch Extension)
qc.e.jal 2147483640

# CHECK: <stdin>:[[@LINE+1]]:11: error: unexpected token
qc.e.j foo@plt
