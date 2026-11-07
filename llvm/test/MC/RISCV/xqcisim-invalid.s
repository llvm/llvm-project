# Xqcisim - Simulaton Hint Instructions
# RUN: llvm-mc --verify=plus -triple riscv32 -mattr=+xqcisim < %s
# RUN: llvm-mc --verify=minus -triple riscv32 -mattr=-xqcisim < %s

# plus-error@+2:14 {{immediate must be an integer in the range [0, 1023]}}
# minus-error@+1 {{invalid instruction}}
qc.psyscalli 1024

# plus-error@+2:1 {{too few operands for instruction}}
# minus-error@+1 {{invalid instruction}}
qc.psyscalli

# plus-error@+2:18 {{unexpected extra operand for instruction}}
# minus-error@+1 {{invalid instruction}}
qc.psyscalli 23, x0

# minus-error@+1:1 {{instruction requires the following: 'Xqcisim' (Qualcomm uC Simulation Hint Extension)}}
qc.psyscalli       1023


# plus-error@+2:11 {{immediate must be an integer in the range [0, 255]}}
# minus-error@+1 {{invalid instruction}}
qc.pputci 256

# plus-error@+2:1 {{too few operands for instruction}}
# minus-error@+1 {{invalid instruction}}
qc.pputci

# plus-error@+2:16 {{unexpected extra operand for instruction}}
# minus-error@+1 {{invalid instruction}}
qc.pputci 200, x8

# minus-error@+1:1 {{instruction requires the following: 'Xqcisim' (Qualcomm uC Simulation Hint Extension)}}
qc.pputci  255


# plus-error@+2:13 {{unexpected extra operand for instruction}}
# minus-error@+1 {{invalid instruction}}
qc.c.ptrace x0

# plus-error@+2:13 {{unexpected extra operand for instruction}}
# minus-error@+1 {{invalid instruction}}
qc.c.ptrace 1

# minus-error@+1:1 {{instruction requires the following: 'Xqcisim' (Qualcomm uC Simulation Hint Extension)}}
qc.c.ptrace


# plus-error@+2:14 {{unexpected extra operand for instruction}}
# minus-error@+1 {{invalid instruction}}
qc.pcoredump 12

# plus-error@+2:14 {{unexpected extra operand for instruction}}
# minus-error@+1 {{invalid instruction}}
qc.pcoredump x4

# minus-error@+1:1 {{instruction requires the following: 'Xqcisim' (Qualcomm uC Simulation Hint Extension)}}
qc.pcoredump


# plus-error@+2:11 {{unexpected extra operand for instruction}}
# minus-error@+1 {{invalid instruction}}
qc.ppregs x1

# plus-error@+2:11 {{unexpected extra operand for instruction}}
# minus-error@+1 {{invalid instruction}}
qc.ppregs 23

# minus-error@+1:1 {{instruction requires the following: 'Xqcisim' (Qualcomm uC Simulation Hint Extension)}}
qc.ppregs


# plus-error@+2:15 {{unexpected extra operand for instruction}}
# minus-error@+1 {{invalid instruction}}
qc.ppreg x10, x2

# plus-error@+2:1 {{too few operands for instruction}}
# minus-error@+1 {{invalid instruction}}
qc.ppreg

# plus-error@+2:10 {{register must be a GPR}}
# minus-error@+1 {{invalid instruction}}
qc.ppreg 23

# minus-error@+1:1 {{instruction requires the following: 'Xqcisim' (Qualcomm uC Simulation Hint Extension)}}
qc.ppreg   a0


# plus-error@+2:14 {{unexpected extra operand for instruction}}
# minus-error@+1 {{invalid instruction}}
qc.pputc x7, x3

# plus-error@+2:1 {{too few operands for instruction}}
# minus-error@+1 {{invalid instruction}}
qc.pputc

# plus-error@+2:10 {{register must be a GPR}}
# minus-error@+1 {{invalid instruction}}
qc.pputc 34

# minus-error@+1:1 {{instruction requires the following: 'Xqcisim' (Qualcomm uC Simulation Hint Extension)}}
qc.pputc   t2


# plus-error@+2:15 {{unexpected extra operand for instruction}}
# minus-error@+1 {{invalid instruction}}
qc.pputs x15, x18

# plus-error@+2:1 {{too few operands for instruction}}
# minus-error@+1 {{invalid instruction}}
qc.pputs

# plus-error@+2:10 {{register must be a GPR}}
# minus-error@+1 {{invalid instruction}}
qc.pputs 45

# minus-error@+1:1 {{instruction requires the following: 'Xqcisim' (Qualcomm uC Simulation Hint Extension)}}
qc.pputs   a5


# plus-error@+2:15 {{unexpected extra operand for instruction}}
# minus-error@+1 {{invalid instruction}}
qc.pexit x26, x23

# plus-error@+2:1 {{too few operands for instruction}}
# minus-error@+1 {{invalid instruction}}
qc.pexit

# plus-error@+2:10 {{register must be a GPR}}
# minus-error@+1 {{invalid instruction}}
qc.pexit 78

# minus-error@+1:1 {{instruction requires the following: 'Xqcisim' (Qualcomm uC Simulation Hint Extension)}}
qc.pexit   s10


# plus-error@+2:18 {{unexpected extra operand for instruction}}
# minus-error@+1 {{invalid instruction}}
qc.psyscall x11, x5

# plus-error@+2:1 {{too few operands for instruction}}
# minus-error@+1 {{invalid instruction}}
qc.psyscall

# plus-error@+2:13 {{register must be a GPR}}
# minus-error@+1 {{invalid instruction}}
qc.psyscall 98

# minus-error@+1:1 {{instruction requires the following: 'Xqcisim' (Qualcomm uC Simulation Hint Extension)}}
qc.psyscall        a1
