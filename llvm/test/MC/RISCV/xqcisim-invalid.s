# Xqcisim - Simulaton Hint Instructions
# RUN: not llvm-mc -triple riscv32 -mattr=+xqcisim < %s 2>&1 \
# RUN:     | FileCheck --match-full-lines -check-prefixes=CHECK-PLUS %s
# RUN: not llvm-mc -triple riscv32 -mattr=-xqcisim < %s 2>&1 \
# RUN:     | FileCheck --match-full-lines -check-prefixes=CHECK-MINUS %s

# CHECK-PLUS: <stdin>:[[@LINE+1]]:14: error: immediate must be an integer in the range [0, 1023]
qc.psyscalli 1024

# CHECK-PLUS: <stdin>:[[@LINE+1]]:1: error: too few operands for instruction
qc.psyscalli

# CHECK-PLUS: <stdin>:[[@LINE+1]]:18: error: unexpected extra operand for instruction
qc.psyscalli 23, x0

# CHECK-MINUS: <stdin>:[[@LINE+1]]:1: error: instruction requires the following: 'Xqcisim' (Qualcomm uC Simulation Hint Extension)
qc.psyscalli       1023


# CHECK-PLUS: <stdin>:[[@LINE+1]]:11: error: immediate must be an integer in the range [0, 255]
qc.pputci 256

# CHECK-PLUS: <stdin>:[[@LINE+1]]:1: error: too few operands for instruction
qc.pputci

# CHECK-PLUS: <stdin>:[[@LINE+1]]:16: error: unexpected extra operand for instruction
qc.pputci 200, x8

# CHECK-MINUS: <stdin>:[[@LINE+1]]:1: error: instruction requires the following: 'Xqcisim' (Qualcomm uC Simulation Hint Extension)
qc.pputci  255


# CHECK-PLUS: <stdin>:[[@LINE+1]]:13: error: unexpected extra operand for instruction
qc.c.ptrace x0

# CHECK-PLUS: <stdin>:[[@LINE+1]]:13: error: unexpected extra operand for instruction
qc.c.ptrace 1

# CHECK-MINUS: <stdin>:[[@LINE+1]]:1: error: instruction requires the following: 'Xqcisim' (Qualcomm uC Simulation Hint Extension)
qc.c.ptrace


# CHECK-PLUS: <stdin>:[[@LINE+1]]:14: error: unexpected extra operand for instruction
qc.pcoredump 12

# CHECK-PLUS: <stdin>:[[@LINE+1]]:14: error: unexpected extra operand for instruction
qc.pcoredump x4

# CHECK-MINUS: <stdin>:[[@LINE+1]]:1: error: instruction requires the following: 'Xqcisim' (Qualcomm uC Simulation Hint Extension)
qc.pcoredump


# CHECK-PLUS: <stdin>:[[@LINE+1]]:11: error: unexpected extra operand for instruction
qc.ppregs x1

# CHECK-PLUS: <stdin>:[[@LINE+1]]:11: error: unexpected extra operand for instruction
qc.ppregs 23

# CHECK-MINUS: <stdin>:[[@LINE+1]]:1: error: instruction requires the following: 'Xqcisim' (Qualcomm uC Simulation Hint Extension)
qc.ppregs


# CHECK-PLUS: <stdin>:[[@LINE+1]]:15: error: unexpected extra operand for instruction
qc.ppreg x10, x2

# CHECK-PLUS: <stdin>:[[@LINE+1]]:1: error: too few operands for instruction
qc.ppreg

# CHECK-PLUS: <stdin>:[[@LINE+1]]:10: error: register must be a GPR
qc.ppreg 23

# CHECK-MINUS: <stdin>:[[@LINE+1]]:1: error: instruction requires the following: 'Xqcisim' (Qualcomm uC Simulation Hint Extension)
qc.ppreg   a0


# CHECK-PLUS: <stdin>:[[@LINE+1]]:14: error: unexpected extra operand for instruction
qc.pputc x7, x3

# CHECK-PLUS: <stdin>:[[@LINE+1]]:1: error: too few operands for instruction
qc.pputc

# CHECK-PLUS: <stdin>:[[@LINE+1]]:10: error: register must be a GPR
qc.pputc 34

# CHECK-MINUS: <stdin>:[[@LINE+1]]:1: error: instruction requires the following: 'Xqcisim' (Qualcomm uC Simulation Hint Extension)
qc.pputc   t2


# CHECK-PLUS: <stdin>:[[@LINE+1]]:15: error: unexpected extra operand for instruction
qc.pputs x15, x18

# CHECK-PLUS: <stdin>:[[@LINE+1]]:1: error: too few operands for instruction
qc.pputs

# CHECK-PLUS: <stdin>:[[@LINE+1]]:10: error: register must be a GPR
qc.pputs 45

# CHECK-MINUS: <stdin>:[[@LINE+1]]:1: error: instruction requires the following: 'Xqcisim' (Qualcomm uC Simulation Hint Extension)
qc.pputs   a5


# CHECK-PLUS: <stdin>:[[@LINE+1]]:15: error: unexpected extra operand for instruction
qc.pexit x26, x23

# CHECK-PLUS: <stdin>:[[@LINE+1]]:1: error: too few operands for instruction
qc.pexit

# CHECK-PLUS: <stdin>:[[@LINE+1]]:10: error: register must be a GPR
qc.pexit 78

# CHECK-MINUS: <stdin>:[[@LINE+1]]:1: error: instruction requires the following: 'Xqcisim' (Qualcomm uC Simulation Hint Extension)
qc.pexit   s10


# CHECK-PLUS: <stdin>:[[@LINE+1]]:18: error: unexpected extra operand for instruction
qc.psyscall x11, x5

# CHECK-PLUS: <stdin>:[[@LINE+1]]:1: error: too few operands for instruction
qc.psyscall

# CHECK-PLUS: <stdin>:[[@LINE+1]]:13: error: register must be a GPR
qc.psyscall 98

# CHECK-MINUS: <stdin>:[[@LINE+1]]:1: error: instruction requires the following: 'Xqcisim' (Qualcomm uC Simulation Hint Extension)
qc.psyscall        a1
