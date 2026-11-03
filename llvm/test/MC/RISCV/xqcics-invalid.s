# Xqcics - Qualcomm uC Conditional Select Extension
# RUN: llvm-mc --verify=plus -triple riscv32 -mattr=+xqcics < %s
# RUN: llvm-mc --verify=minus -triple riscv32 -mattr=-xqcics < %s

# plus-error@+2:14 {{register must be a GPR excluding zero (x0)}}
# minus-error@+1:1 {{invalid instruction}}
qc.selecteqi 9, 15, x4, x3

# plus-error@+2:24 {{too few operands for instruction}}
# minus-error@+1:1 {{invalid instruction}}
qc.selecteqi x9, 15, x4

# plus-error@+2:18 {{immediate must be an integer in the range [-16, 15]}}
# minus-error@+1 {{invalid instruction}}
qc.selecteqi x9, 16, x4, x3

# minus-error@+1:1 {{instruction requires the following: 'Xqcics' (Qualcomm uC Conditional Select Extension)}}
qc.selecteqi x9, 15, x4, x3


# plus-error@+2:14 {{register must be a GPR excluding zero (x0)}}
# minus-error@+1:1 {{invalid instruction}}
qc.selectieq 8, x4, x3, 12

# plus-error@+2:24 {{too few operands for instruction}}
# minus-error@+1:1 {{invalid instruction}}
qc.selectieq x8, x4, x3

# plus-error@+2:26 {{immediate must be an integer in the range [-16, 15]}}
# minus-error@+1 {{invalid instruction}}
qc.selectieq x8, x4, x3, 17

# minus-error@+1:1 {{instruction requires the following: 'Xqcics' (Qualcomm uC Conditional Select Extension)}}
qc.selectieq x8, x4, x3, 12


# plus-error@+2:15 {{register must be a GPR excluding zero (x0)}}
# minus-error@+1:1 {{invalid instruction}}
qc.selectieqi 9, 11, x3, 12

# plus-error@+2:25 {{too few operands for instruction}}
# minus-error@+1:1 {{invalid instruction}}
qc.selectieqi x9, 11, x3

# plus-error@+2:19 {{immediate must be an integer in the range [-16, 15]}}
# minus-error@+1 {{invalid instruction}}
qc.selectieqi x9, 16, x3, 12

# plus-error@+2:27 {{immediate must be an integer in the range [-16, 15]}}
# minus-error@+1 {{invalid instruction}}
qc.selectieqi x9, 11, x3, 18

# minus-error@+1:1 {{instruction requires the following: 'Xqcics' (Qualcomm uC Conditional Select Extension)}}
qc.selectieqi x9, 11, x3, 12


# plus-error@+2:15 {{register must be a GPR excluding zero (x0)}}
# minus-error@+1:1 {{invalid instruction}}
qc.selectiieq 9, x3, 11, 12

# plus-error@+2:25 {{too few operands for instruction}}
# minus-error@+1:1 {{invalid instruction}}
qc.selectiieq x9, x3, 11

# plus-error@+2:23 {{immediate must be an integer in the range [-16, 15]}}
# minus-error@+1 {{invalid instruction}}
qc.selectiieq x9, x3, 16, 12

# plus-error@+2:27 {{immediate must be an integer in the range [-16, 15]}}
# minus-error@+1 {{invalid instruction}}
qc.selectiieq x9, x3, 11, 17

# minus-error@+1:1 {{instruction requires the following: 'Xqcics' (Qualcomm uC Conditional Select Extension)}}
qc.selectiieq x9, x3, 11, 12


# plus-error@+2:15 {{register must be a GPR excluding zero (x0)}}
# minus-error@+1:1 {{invalid instruction}}
qc.selectiine 8, x3, 10, 11

# plus-error@+2:25 {{too few operands for instruction}}
# minus-error@+1:1 {{invalid instruction}}
qc.selectiine x8, x3, 10

# plus-error@+2:23 {{immediate must be an integer in the range [-16, 15]}}
# minus-error@+1 {{invalid instruction}}
qc.selectiine x8, x3, 16, 11

# plus-error@+2:27 {{immediate must be an integer in the range [-16, 15]}}
# minus-error@+1 {{invalid instruction}}
qc.selectiine x8, x3, 12, 18

# minus-error@+1:1 {{instruction requires the following: 'Xqcics' (Qualcomm uC Conditional Select Extension)}}
qc.selectiine x8, x3, 10, 11


# plus-error@+2:14 {{register must be a GPR excluding zero (x0)}}
# minus-error@+1:1 {{invalid instruction}}
qc.selectine 8, x3, x4, 11

# plus-error@+2:24 {{too few operands for instruction}}
# minus-error@+1:1 {{invalid instruction}}
qc.selectine x8, x3, x4

# plus-error@+2:26 {{immediate must be an integer in the range [-16, 15]}}
# minus-error@+1 {{invalid instruction}}
qc.selectine x8, x3, x4, 16

# minus-error@+1:1 {{instruction requires the following: 'Xqcics' (Qualcomm uC Conditional Select Extension)}}
qc.selectine x8, x3, x4, 11


# plus-error@+2:15 {{register must be a GPR excluding zero (x0)}}
# minus-error@+1:1 {{invalid instruction}}
qc.selectinei 8, 11, x3, 12

# plus-error@+2:25 {{too few operands for instruction}}
# minus-error@+1:1 {{invalid instruction}}
qc.selectinei x8, 11, x3

# plus-error@+2:19 {{immediate must be an integer in the range [-16, 15]}}
# minus-error@+1 {{invalid instruction}}
qc.selectinei x8, 16, x3, 12

# plus-error@+2:27 {{immediate must be an integer in the range [-16, 15]}}
# minus-error@+1 {{invalid instruction}}
qc.selectinei x8, 11, x3, 18

# minus-error@+1:1 {{instruction requires the following: 'Xqcics' (Qualcomm uC Conditional Select Extension)}}
qc.selectinei x8, 11, x3, 12


# plus-error@+2:14 {{register must be a GPR excluding zero (x0)}}
# minus-error@+1:1 {{invalid instruction}}
qc.selectnei 8, 11, x3, x5

# plus-error@+2:24 {{too few operands for instruction}}
# minus-error@+1:1 {{invalid instruction}}
qc.selectnei x8, 11, x3

# plus-error@+2:18 {{immediate must be an integer in the range [-16, 15]}}
# minus-error@+1 {{invalid instruction}}
qc.selectnei x8, 16, x3, x5

# minus-error@+1:1 {{instruction requires the following: 'Xqcics' (Qualcomm uC Conditional Select Extension)}}
qc.selectnei x8, 11, x3, x5

