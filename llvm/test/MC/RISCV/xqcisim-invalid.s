# Xqcisim - Simulaton Hint Instructions
# RUN: llvm-mc --verify=plus -triple riscv32 -mattr=+xqcisim < %s
# RUN: llvm-mc --verify=minus -triple riscv32 -mattr=-xqcisim < %s

qc.psyscalli 1024
# plus-error@-1:14 {{immediate must be an integer in the range [0, 1023]}}
# minus-error@-2 {{invalid instruction}}

qc.psyscalli
# plus-error@-1:1 {{too few operands for instruction}}
# minus-error@-2 {{invalid instruction}}

qc.psyscalli 23, x0
# plus-error@-1:18 {{unexpected extra operand for instruction}}
# minus-error@-2 {{invalid instruction}}

qc.psyscalli       1023 # minus-error:1 {{instruction requires the following: 'Xqcisim' (Qualcomm uC Simulation Hint Extension)}}


qc.pputci 256
# plus-error@-1:11 {{immediate must be an integer in the range [0, 255]}}
# minus-error@-2 {{invalid instruction}}

qc.pputci
# plus-error@-1:1 {{too few operands for instruction}}
# minus-error@-2 {{invalid instruction}}

qc.pputci 200, x8
# plus-error@-1:16 {{unexpected extra operand for instruction}}
# minus-error@-2 {{invalid instruction}}

qc.pputci  255 # minus-error:1 {{instruction requires the following: 'Xqcisim' (Qualcomm uC Simulation Hint Extension)}}


qc.c.ptrace x0
# plus-error@-1:13 {{unexpected extra operand for instruction}}
# minus-error@-2 {{invalid instruction}}

qc.c.ptrace 1
# plus-error@-1:13 {{unexpected extra operand for instruction}}
# minus-error@-2 {{invalid instruction}}

qc.c.ptrace # minus-error:1 {{instruction requires the following: 'Xqcisim' (Qualcomm uC Simulation Hint Extension)}}


qc.pcoredump 12
# plus-error@-1:14 {{unexpected extra operand for instruction}}
# minus-error@-2 {{invalid instruction}}

qc.pcoredump x4
# plus-error@-1:14 {{unexpected extra operand for instruction}}
# minus-error@-2 {{invalid instruction}}

qc.pcoredump # minus-error:1 {{instruction requires the following: 'Xqcisim' (Qualcomm uC Simulation Hint Extension)}}


qc.ppregs x1
# plus-error@-1:11 {{unexpected extra operand for instruction}}
# minus-error@-2 {{invalid instruction}}

qc.ppregs 23
# plus-error@-1:11 {{unexpected extra operand for instruction}}
# minus-error@-2 {{invalid instruction}}

qc.ppregs # minus-error:1 {{instruction requires the following: 'Xqcisim' (Qualcomm uC Simulation Hint Extension)}}


qc.ppreg x10, x2
# plus-error@-1:15 {{unexpected extra operand for instruction}}
# minus-error@-2 {{invalid instruction}}

qc.ppreg
# plus-error@-1:1 {{too few operands for instruction}}
# minus-error@-2 {{invalid instruction}}

qc.ppreg 23
# plus-error@-1:10 {{register must be a GPR}}
# minus-error@-2 {{invalid instruction}}

qc.ppreg   a0 # minus-error:1 {{instruction requires the following: 'Xqcisim' (Qualcomm uC Simulation Hint Extension)}}


qc.pputc x7, x3
# plus-error@-1:14 {{unexpected extra operand for instruction}}
# minus-error@-2 {{invalid instruction}}

qc.pputc
# plus-error@-1:1 {{too few operands for instruction}}
# minus-error@-2 {{invalid instruction}}

qc.pputc 34
# plus-error@-1:10 {{register must be a GPR}}
# minus-error@-2 {{invalid instruction}}

qc.pputc   t2 # minus-error:1 {{instruction requires the following: 'Xqcisim' (Qualcomm uC Simulation Hint Extension)}}


qc.pputs x15, x18
# plus-error@-1:15 {{unexpected extra operand for instruction}}
# minus-error@-2 {{invalid instruction}}

qc.pputs
# plus-error@-1:1 {{too few operands for instruction}}
# minus-error@-2 {{invalid instruction}}

qc.pputs 45
# plus-error@-1:10 {{register must be a GPR}}
# minus-error@-2 {{invalid instruction}}

qc.pputs   a5 # minus-error:1 {{instruction requires the following: 'Xqcisim' (Qualcomm uC Simulation Hint Extension)}}


qc.pexit x26, x23
# plus-error@-1:15 {{unexpected extra operand for instruction}}
# minus-error@-2 {{invalid instruction}}

qc.pexit
# plus-error@-1:1 {{too few operands for instruction}}
# minus-error@-2 {{invalid instruction}}

qc.pexit 78
# plus-error@-1:10 {{register must be a GPR}}
# minus-error@-2 {{invalid instruction}}

qc.pexit   s10 # minus-error:1 {{instruction requires the following: 'Xqcisim' (Qualcomm uC Simulation Hint Extension)}}


qc.psyscall x11, x5
# plus-error@-1:18 {{unexpected extra operand for instruction}}
# minus-error@-2 {{invalid instruction}}

qc.psyscall
# plus-error@-1:1 {{too few operands for instruction}}
# minus-error@-2 {{invalid instruction}}

qc.psyscall 98
# plus-error@-1:13 {{register must be a GPR}}
# minus-error@-2 {{invalid instruction}}

qc.psyscall        a1 # minus-error:1 {{instruction requires the following: 'Xqcisim' (Qualcomm uC Simulation Hint Extension)}}
