# Xqcics - Qualcomm uC Conditional Select Extension
# RUN: llvm-mc --verify=plus -triple riscv32 -mattr=+xqcics < %s
# RUN: llvm-mc --verify=minus -triple riscv32 -mattr=-xqcics < %s

qc.selecteqi 9, 15, x4, x3
# plus-error@-1:14 {{register must be a GPR excluding zero (x0)}}
# minus-error@-2:1 {{invalid instruction}}

qc.selecteqi x9, 15, x4
# plus-error@-1:24 {{too few operands for instruction}}
# minus-error@-2:1 {{invalid instruction}}

qc.selecteqi x9, 16, x4, x3
# plus-error@-1:18 {{immediate must be an integer in the range [-16, 15]}}
# minus-error@-2 {{invalid instruction}}

qc.selecteqi x9, 15, x4, x3 # minus-error:1 {{instruction requires the following: 'Xqcics' (Qualcomm uC Conditional Select Extension)}}


qc.selectieq 8, x4, x3, 12
# plus-error@-1:14 {{register must be a GPR excluding zero (x0)}}
# minus-error@-2:1 {{invalid instruction}}

qc.selectieq x8, x4, x3
# plus-error@-1:24 {{too few operands for instruction}}
# minus-error@-2:1 {{invalid instruction}}

qc.selectieq x8, x4, x3, 17
# plus-error@-1:26 {{immediate must be an integer in the range [-16, 15]}}
# minus-error@-2 {{invalid instruction}}

qc.selectieq x8, x4, x3, 12 # minus-error:1 {{instruction requires the following: 'Xqcics' (Qualcomm uC Conditional Select Extension)}}


qc.selectieqi 9, 11, x3, 12
# plus-error@-1:15 {{register must be a GPR excluding zero (x0)}}
# minus-error@-2:1 {{invalid instruction}}

qc.selectieqi x9, 11, x3
# plus-error@-1:25 {{too few operands for instruction}}
# minus-error@-2:1 {{invalid instruction}}

qc.selectieqi x9, 16, x3, 12
# plus-error@-1:19 {{immediate must be an integer in the range [-16, 15]}}
# minus-error@-2 {{invalid instruction}}

qc.selectieqi x9, 11, x3, 18
# plus-error@-1:27 {{immediate must be an integer in the range [-16, 15]}}
# minus-error@-2 {{invalid instruction}}

qc.selectieqi x9, 11, x3, 12 # minus-error:1 {{instruction requires the following: 'Xqcics' (Qualcomm uC Conditional Select Extension)}}


qc.selectiieq 9, x3, 11, 12
# plus-error@-1:15 {{register must be a GPR excluding zero (x0)}}
# minus-error@-2:1 {{invalid instruction}}

qc.selectiieq x9, x3, 11
# plus-error@-1:25 {{too few operands for instruction}}
# minus-error@-2:1 {{invalid instruction}}

qc.selectiieq x9, x3, 16, 12
# plus-error@-1:23 {{immediate must be an integer in the range [-16, 15]}}
# minus-error@-2 {{invalid instruction}}

qc.selectiieq x9, x3, 11, 17
# plus-error@-1:27 {{immediate must be an integer in the range [-16, 15]}}
# minus-error@-2 {{invalid instruction}}

qc.selectiieq x9, x3, 11, 12 # minus-error:1 {{instruction requires the following: 'Xqcics' (Qualcomm uC Conditional Select Extension)}}


qc.selectiine 8, x3, 10, 11
# plus-error@-1:15 {{register must be a GPR excluding zero (x0)}}
# minus-error@-2:1 {{invalid instruction}}

qc.selectiine x8, x3, 10
# plus-error@-1:25 {{too few operands for instruction}}
# minus-error@-2:1 {{invalid instruction}}

qc.selectiine x8, x3, 16, 11
# plus-error@-1:23 {{immediate must be an integer in the range [-16, 15]}}
# minus-error@-2 {{invalid instruction}}

qc.selectiine x8, x3, 12, 18
# plus-error@-1:27 {{immediate must be an integer in the range [-16, 15]}}
# minus-error@-2 {{invalid instruction}}

qc.selectiine x8, x3, 10, 11 # minus-error:1 {{instruction requires the following: 'Xqcics' (Qualcomm uC Conditional Select Extension)}}


qc.selectine 8, x3, x4, 11
# plus-error@-1:14 {{register must be a GPR excluding zero (x0)}}
# minus-error@-2:1 {{invalid instruction}}

qc.selectine x8, x3, x4
# plus-error@-1:24 {{too few operands for instruction}}
# minus-error@-2:1 {{invalid instruction}}

qc.selectine x8, x3, x4, 16
# plus-error@-1:26 {{immediate must be an integer in the range [-16, 15]}}
# minus-error@-2 {{invalid instruction}}

qc.selectine x8, x3, x4, 11 # minus-error:1 {{instruction requires the following: 'Xqcics' (Qualcomm uC Conditional Select Extension)}}


qc.selectinei 8, 11, x3, 12
# plus-error@-1:15 {{register must be a GPR excluding zero (x0)}}
# minus-error@-2:1 {{invalid instruction}}

qc.selectinei x8, 11, x3
# plus-error@-1:25 {{too few operands for instruction}}
# minus-error@-2:1 {{invalid instruction}}

qc.selectinei x8, 16, x3, 12
# plus-error@-1:19 {{immediate must be an integer in the range [-16, 15]}}
# minus-error@-2 {{invalid instruction}}

qc.selectinei x8, 11, x3, 18
# plus-error@-1:27 {{immediate must be an integer in the range [-16, 15]}}
# minus-error@-2 {{invalid instruction}}

qc.selectinei x8, 11, x3, 12 # minus-error:1 {{instruction requires the following: 'Xqcics' (Qualcomm uC Conditional Select Extension)}}


qc.selectnei 8, 11, x3, x5
# plus-error@-1:14 {{register must be a GPR excluding zero (x0)}}
# minus-error@-2:1 {{invalid instruction}}

qc.selectnei x8, 11, x3
# plus-error@-1:24 {{too few operands for instruction}}
# minus-error@-2:1 {{invalid instruction}}

qc.selectnei x8, 16, x3, x5
# plus-error@-1:18 {{immediate must be an integer in the range [-16, 15]}}
# minus-error@-2 {{invalid instruction}}

qc.selectnei x8, 11, x3, x5 # minus-error:1 {{instruction requires the following: 'Xqcics' (Qualcomm uC Conditional Select Extension)}}

