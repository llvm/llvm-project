# Xqciint - Qualcomm uC Interrupts extension
# RUN: llvm-mc --verify=plus -triple riscv32 -mattr=+xqciint < %s
# RUN: llvm-mc --verify=minus -triple riscv32 -mattr=-xqciint < %s

# plus-error@+2:12 {{immediate must be an integer in the range [0, 1023]}}
# minus-error@+1:1 {{invalid instruction}}
qc.setinti 1025

# plus-error@+2:16 {{unexpected extra operand for instruction}}
# minus-error@+1:1 {{invalid instruction}}
qc.setinti 11, 12

# plus-error@+2:1 {{too few operands for instruction}}
# minus-error@+1:1 {{invalid instruction}}
qc.setinti

# minus-error@+1:1 {{instruction requires the following: 'Xqciint' (Qualcomm uC Interrupts Extension)}}
qc.setinti   10


# plus-error@+2:12 {{immediate must be an integer in the range [0, 1023]}}
# minus-error@+1 {{invalid instruction}}
qc.clrinti 2000

# plus-error@+2:16 {{unexpected extra operand for instruction}}
# minus-error@+1:1 {{invalid instruction}}
qc.clrinti 22, x4

# plus-error@+2:1 {{too few operands for instruction}}
# minus-error@+1:1 {{invalid instruction}}
qc.clrinti

# minus-error@+1:1 {{instruction requires the following: 'Xqciint' (Qualcomm uC Interrupts Extension)}}
qc.clrinti 8


# plus-error@+2:13 {{register must be a GPR excluding zero (x0)}}
# minus-error@+1:1 {{invalid instruction}}
qc.c.clrint 22

# plus-error@+2:1 {{too few operands for instruction}}
# minus-error@+1:1 {{invalid instruction}}
qc.c.clrint

# minus-error@+1:1 {{instruction requires the following: 'Xqciint' (Qualcomm uC Interrupts Extension)}}
qc.c.clrint x8


# plus-error@+2:9 {{unexpected extra operand for instruction}}
# minus-error@+1:1 {{invalid instruction}}
qc.c.di 22

# minus-error@+1:1 {{instruction requires the following: 'Xqciint' (Qualcomm uC Interrupts Extension)}}
qc.c.di


# plus-error@+2:10 {{register must be a GPR excluding zero (x0)}}
# minus-error@+1:1 {{invalid instruction}}
qc.c.dir 22

# plus-error@+2:1 {{too few operands for instruction}}
# minus-error@+1:1 {{invalid instruction}}
qc.c.dir

# minus-error@+1:1 {{instruction requires the following: 'Xqciint' (Qualcomm uC Interrupts Extension)}}
qc.c.dir x8


# plus-error@+2:9 {{unexpected extra operand for instruction}}
# minus-error@+1:1 {{invalid instruction}}
qc.c.ei 22

# minus-error@+1:1 {{instruction requires the following: 'Xqciint' (Qualcomm uC Interrupts Extension)}}
qc.c.ei


# plus-error@+2:10 {{register must be a GPR excluding zero (x0)}}
# minus-error@+1:1 {{invalid instruction}}
qc.c.eir 22

# plus-error@+2:1 {{too few operands for instruction}}
# minus-error@+1:1 {{invalid instruction}}
qc.c.eir

# minus-error@+1:1 {{instruction requires the following: 'Xqciint' (Qualcomm uC Interrupts Extension)}}
qc.c.eir x8


# plus-error@+2:19 {{unexpected extra operand for instruction}}
# minus-error@+1:1 {{invalid instruction}}
qc.c.mienter.nest 22

# minus-error@+1:1 {{instruction requires the following: 'Xqciint' (Qualcomm uC Interrupts Extension)}}
qc.c.mienter.nest


# plus-error@+2:14 {{unexpected extra operand for instruction}}
# minus-error@+1:1 {{invalid instruction}}
qc.c.mienter 22

# minus-error@+1:1 {{instruction requires the following: 'Xqciint' (Qualcomm uC Interrupts Extension)}}
qc.c.mienter


# plus-error@+2:17 {{unexpected extra operand for instruction}}
# minus-error@+1:1 {{invalid instruction}}
qc.c.mileaveret 22

# minus-error@+1:1 {{instruction requires the following: 'Xqciint' (Qualcomm uC Interrupts Extension)}}
qc.c.mileaveret


# plus-error@+2:13 {{register must be a GPR excluding zero (x0)}}
# minus-error@+1:1 {{invalid instruction}}
qc.c.setint 22

# plus-error@+2:1 {{too few operands for instruction}}
# minus-error@+1:1 {{invalid instruction}}
qc.c.setint

# minus-error@+1:1 {{instruction requires the following: 'Xqciint' (Qualcomm uC Interrupts Extension)}}
qc.c.setint x8


# plus-error@+2:11 {{unexpected extra operand for instruction}}
# minus-error@+1:1 {{invalid instruction}}
qc.c.mret x8

# minus-error@+1:1 {{instruction requires the following: 'Xqciint' (Qualcomm uC Interrupts Extension)}}
qc.c.mret


# plus-error@+2:12 {{unexpected extra operand for instruction}}
# minus-error@+1:1 {{invalid instruction}}
qc.c.mnret 10

# minus-error@+1:1 {{instruction requires the following: 'Xqciint' (Qualcomm uC Interrupts Extension)}}
qc.c.mnret
