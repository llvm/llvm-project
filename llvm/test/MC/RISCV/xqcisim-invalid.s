# Xqcisim - Simulaton Hint Instructions
# RUN: not llvm-mc -triple riscv32 -mattr=+experimental-xqcisim < %s 2>&1 \
# RUN:     | FileCheck -check-prefixes=CHECK,CHECK-PLUS %s
# RUN: not llvm-mc -triple riscv32 -mattr=-experimental-xqcisim < %s 2>&1 \
# RUN:     | FileCheck -check-prefixes=CHECK,CHECK-MINUS %s

# CHECK-PLUS: :[[@LINE+1]]:14: error: immediate must be an integer in the range [0, 1023]
qc.psyscalli 1024

# CHECK: :[[@LINE+1]]:1: error: too few operands for instruction
qc.psyscalli

# CHECK: :[[@LINE+1]]:18: error: invalid operand for instruction
qc.psyscalli 23, x0

# CHECK-MINUS: :[[@LINE+1]]:1: error: instruction requires the following: 'Xqcisim' (Qualcomm uC Simulation Hint Extension)
qc.psyscalli       1023


# CHECK-PLUS: :[[@LINE+1]]:11: error: immediate must be an integer in the range [0, 255]
qc.pputci 256

# CHECK: :[[@LINE+1]]:1: error: too few operands for instruction
qc.pputci

# CHECK: :[[@LINE+1]]:16: error: invalid operand for instruction
qc.pputci 200, x8

# CHECK-MINUS: :[[@LINE+1]]:1: error: instruction requires the following: 'Xqcisim' (Qualcomm uC Simulation Hint Extension)
qc.pputci  255


# CHECK: :[[@LINE+1]]:13: error: invalid operand for instruction
qc.c.ptrace x0

# CHECK: :[[@LINE+1]]:13: error: invalid operand for instruction
qc.c.ptrace 1

# CHECK-MINUS: :[[@LINE+1]]:1: error: instruction requires the following: 'Xqcisim' (Qualcomm uC Simulation Hint Extension)
qc.c.ptrace


# CHECK: :[[@LINE+1]]:14: error: invalid operand for instruction
qc.pcoredump 12

# CHECK: :[[@LINE+1]]:14: error: invalid operand for instruction
qc.pcoredump x4

# CHECK-MINUS: :[[@LINE+1]]:1: error: instruction requires the following: 'Xqcisim' (Qualcomm uC Simulation Hint Extension)
qc.pcoredump


# CHECK: :[[@LINE+1]]:11: error: invalid operand for instruction
qc.ppregs x1

# CHECK: :[[@LINE+1]]:11: error: invalid operand for instruction
qc.ppregs 23

# CHECK-MINUS: :[[@LINE+1]]:1: error: instruction requires the following: 'Xqcisim' (Qualcomm uC Simulation Hint Extension)
qc.ppregs


# CHECK: :[[@LINE+1]]:15: error: invalid operand for instruction
qc.ppreg x10, x2

# CHECK: :[[@LINE+1]]:1: error: too few operands for instruction
qc.ppreg

# CHECK: :[[@LINE+1]]:10: error: invalid operand for instruction
qc.ppreg 23

# CHECK-MINUS: :[[@LINE+1]]:1: error: instruction requires the following: 'Xqcisim' (Qualcomm uC Simulation Hint Extension)
qc.ppreg   a0


# CHECK: :[[@LINE+1]]:14: error: invalid operand for instruction
qc.pputc x7, x3

# CHECK: :[[@LINE+1]]:1: error: too few operands for instruction
qc.pputc

# CHECK: :[[@LINE+1]]:10: error: invalid operand for instruction
qc.pputc 34

# CHECK-MINUS: :[[@LINE+1]]:1: error: instruction requires the following: 'Xqcisim' (Qualcomm uC Simulation Hint Extension)
qc.pputc   t2


# CHECK: :[[@LINE+1]]:15: error: invalid operand for instruction
qc.pputs x15, x18

# CHECK: :[[@LINE+1]]:1: error: too few operands for instruction
qc.pputs

# CHECK: :[[@LINE+1]]:10: error: invalid operand for instruction
qc.pputs 45

# CHECK-MINUS: :[[@LINE+1]]:1: error: instruction requires the following: 'Xqcisim' (Qualcomm uC Simulation Hint Extension)
qc.pputs   a5


# CHECK: :[[@LINE+1]]:15: error: invalid operand for instruction
qc.pexit x26, x23

# CHECK: :[[@LINE+1]]:1: error: too few operands for instruction
qc.pexit

# CHECK: :[[@LINE+1]]:10: error: invalid operand for instruction
qc.pexit 78

# CHECK-MINUS: :[[@LINE+1]]:1: error: instruction requires the following: 'Xqcisim' (Qualcomm uC Simulation Hint Extension)
qc.pexit   s10


# CHECK: :[[@LINE+1]]:18: error: invalid operand for instruction
qc.psyscall x11, x5

# CHECK: :[[@LINE+1]]:1: error: too few operands for instruction
qc.psyscall

# CHECK: :[[@LINE+1]]:13: error: invalid operand for instruction
qc.psyscall 98

# CHECK-MINUS: :[[@LINE+1]]:1: error: instruction requires the following: 'Xqcisim' (Qualcomm uC Simulation Hint Extension)
qc.psyscall        a1
