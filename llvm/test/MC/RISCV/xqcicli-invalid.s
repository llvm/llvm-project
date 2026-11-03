# Xqcicli - Qualcomm uC Conditional Load Immediate Instructions
# RUN: llvm-mc --verify=plus -triple riscv32 -mattr=+xqcicli < %s
# RUN: llvm-mc --verify=minus -triple riscv32 -mattr=-xqcicli < %s

# plus-error@+2:9 {{register must be a GPR excluding zero (x0)}}
# minus-error@+1:1 {{invalid instruction}}
qc.lieq x0, x4, x6, 10

# plus-error@+2:13 {{register must be a GPR excluding zero (x0)}}
# minus-error@+1:1 {{invalid instruction}}
qc.lieq x2, x0, x6, 10

# plus-error@+2:17 {{register must be a GPR excluding zero (x0)}}
# minus-error@+1:1 {{invalid instruction}}
qc.lieq x2, x4, x0, 10

# plus-error@+2:19 {{too few operands for instruction}}
# minus-error@+1:1 {{invalid instruction}}
qc.lieq x2, x4, x6

# plus-error@+2:21 {{immediate must be an integer in the range [-16, 15]}}
# minus-error@+1 {{invalid instruction}}
qc.lieq x2, x4, x6, 40

# minus-error@+1:1 {{instruction requires the following: 'Xqcicli' (Qualcomm uC Conditional Load Immediate Extension)}}
qc.lieq x2, x4, x6, 10


# plus-error@+2:9 {{register must be a GPR excluding zero (x0)}}
# minus-error@+1:1 {{invalid instruction}}
qc.lige x0, x8, x20, 2

# plus-error@+2:13 {{register must be a GPR excluding zero (x0)}}
# minus-error@+1:1 {{invalid instruction}}
qc.lige x4, x0, x20, 2

# plus-error@+2:17 {{register must be a GPR excluding zero (x0)}}
# minus-error@+1:1 {{invalid instruction}}
qc.lige x4, x8, x0, 2

# plus-error@+2:20 {{too few operands for instruction}}
# minus-error@+1:1 {{invalid instruction}}
qc.lige x4, x8, x20

# plus-error@+2:22 {{immediate must be an integer in the range [-16, 15]}}
# minus-error@+1 {{invalid instruction}}
qc.lige x4, x8, x20, -18

# minus-error@+1:1 {{instruction requires the following: 'Xqcicli' (Qualcomm uC Conditional Load Immediate Extension)}}
qc.lige x4, x8, x20, 2


# plus-error@+2:9 {{register must be a GPR excluding zero (x0)}}
# minus-error@+1:1 {{invalid instruction}}
qc.lilt x0, x9, x10, 3

# plus-error@+2:14 {{register must be a GPR excluding zero (x0)}}
# minus-error@+1:1 {{invalid instruction}}
qc.lilt x19, x0, x10, 3

# plus-error@+2:18 {{register must be a GPR excluding zero (x0)}}
# minus-error@+1:1 {{invalid instruction}}
qc.lilt x19, x9, x0, 3

# plus-error@+2:21 {{too few operands for instruction}}
# minus-error@+1:1 {{invalid instruction}}
qc.lilt x19, x9, x10

# plus-error@+2:23 {{immediate must be an integer in the range [-16, 15]}}
# minus-error@+1 {{invalid instruction}}
qc.lilt x19, x9, x10, 39

# minus-error@+1:1 {{instruction requires the following: 'Xqcicli' (Qualcomm uC Conditional Load Immediate Extension)}}
qc.lilt x19, x9, x10, 3


# plus-error@+2:9 {{register must be a GPR excluding zero (x0)}}
# minus-error@+1:1 {{invalid instruction}}
qc.line x0, x14, x6, 10

# plus-error@+2:14 {{register must be a GPR excluding zero (x0)}}
# minus-error@+1:1 {{invalid instruction}}
qc.line x18, x0, x6, 10

# plus-error@+2:19 {{register must be a GPR excluding zero (x0)}}
# minus-error@+1:1 {{invalid instruction}}
qc.line x18, x14, x0, 10

# plus-error@+2:21 {{too few operands for instruction}}
# minus-error@+1:1 {{invalid instruction}}
qc.line x18, x14, x6

# plus-error@+2:23 {{immediate must be an integer in the range [-16, 15]}}
# minus-error@+1 {{invalid instruction}}
qc.line x18, x14, x6, 100

# minus-error@+1:1 {{instruction requires the following: 'Xqcicli' (Qualcomm uC Conditional Load Immediate Extension)}}
qc.line x18, x14, x6, 10


# plus-error@+2:10 {{register must be a GPR excluding zero (x0)}}
# minus-error@+1:1 {{invalid instruction}}
qc.ligeu x0, x4, x6, 10

# plus-error@+2:14 {{register must be a GPR excluding zero (x0)}}
# minus-error@+1:1 {{invalid instruction}}
qc.ligeu x2, x0, x6, 10

# plus-error@+2:18 {{register must be a GPR excluding zero (x0)}}
# minus-error@+1:1 {{invalid instruction}}
qc.ligeu x2, x4, x0, 10

# plus-error@+2:20 {{too few operands for instruction}}
# minus-error@+1:1 {{invalid instruction}}
qc.ligeu x2, x4, x6

# plus-error@+2:22 {{immediate must be an integer in the range [-16, 15]}}
# minus-error@+1 {{invalid instruction}}
qc.ligeu x2, x4, x6, 70

# minus-error@+1:1 {{instruction requires the following: 'Xqcicli' (Qualcomm uC Conditional Load Immediate Extension)}}
qc.ligeu x2, x4, x6, 10


# plus-error@+2:10 {{register must be a GPR excluding zero (x0)}}
# minus-error@+1:1 {{invalid instruction}}
qc.liltu x0, x19, x12, 13

# plus-error@+2:14 {{register must be a GPR excluding zero (x0)}}
# minus-error@+1:1 {{invalid instruction}}
qc.liltu x1, x0, x12, 13

# plus-error@+2:19 {{register must be a GPR excluding zero (x0)}}
# minus-error@+1:1 {{invalid instruction}}
qc.liltu x1, x19, x0, 13

# plus-error@+2:22 {{too few operands for instruction}}
# minus-error@+1:1 {{invalid instruction}}
qc.liltu x1, x19, x12

# plus-error@+2:24 {{immediate must be an integer in the range [-16, 15]}}
# minus-error@+1 {{invalid instruction}}
qc.liltu x1, x19, x12, 73

# minus-error@+1:1 {{instruction requires the following: 'Xqcicli' (Qualcomm uC Conditional Load Immediate Extension)}}
qc.liltu x1, x19, x12, 13


# plus-error@+2:10 {{register must be a GPR excluding zero (x0)}}
# minus-error@+1:1 {{invalid instruction}}
qc.lieqi x0, x1, 15, 12

# plus-error@+2:14 {{register must be a GPR excluding zero (x0)}}
# minus-error@+1:1 {{invalid instruction}}
qc.lieqi x7, x0, 15, 12

# plus-error@+2:20 {{too few operands for instruction}}
# minus-error@+1:1 {{invalid instruction}}
qc.lieqi x7, x1, 15

# plus-error@+2:18 {{immediate must be an integer in the range [-16, 15]}}
# minus-error@+1 {{invalid instruction}}
qc.lieqi x7, x1, 25, 12

# plus-error@+2:22 {{immediate must be an integer in the range [-16, 15]}}
# minus-error@+1 {{invalid instruction}}
qc.lieqi x7, x1, 15, -22

# minus-error@+1:1 {{instruction requires the following: 'Xqcicli' (Qualcomm uC Conditional Load Immediate Extension)}}
qc.lieqi x7, x1, 15, 12


# plus-error@+2:10 {{register must be a GPR excluding zero (x0)}}
# minus-error@+1:1 {{invalid instruction}}
qc.ligei x0, x11, -4, 9

# plus-error@+2:15 {{register must be a GPR excluding zero (x0)}}
# minus-error@+1:1 {{invalid instruction}}
qc.ligei x17, x0, -4, 9

# plus-error@+2:22 {{too few operands for instruction}}
# minus-error@+1:1 {{invalid instruction}}
qc.ligei x17, x11, -4

# plus-error@+2:20 {{immediate must be an integer in the range [-16, 15]}}
# minus-error@+1 {{invalid instruction}}
qc.ligei x17, x11, -24, 9

# plus-error@+2:24 {{immediate must be an integer in the range [-16, 15]}}
# minus-error@+1 {{invalid instruction}}
qc.ligei x17, x11, -4, 59

# minus-error@+1:1 {{instruction requires the following: 'Xqcicli' (Qualcomm uC Conditional Load Immediate Extension)}}
qc.ligei x17, x11, -4, 9


# plus-error@+2:10 {{register must be a GPR excluding zero (x0)}}
# minus-error@+1:1 {{invalid instruction}}
qc.lilti x0, x11, -14, 2

# plus-error@+2:14 {{register must be a GPR excluding zero (x0)}}
# minus-error@+1:1 {{invalid instruction}}
qc.lilti x9, x0, -14, 2

# plus-error@+2:22 {{too few operands for instruction}}
# minus-error@+1:1 {{invalid instruction}}
qc.lilti x9, x11, -14

# plus-error@+2:19 {{immediate must be an integer in the range [-16, 15]}}
# minus-error@+1 {{invalid instruction}}
qc.lilti x9, x11, -84, 2

# plus-error@+2:24 {{immediate must be an integer in the range [-16, 15]}}
# minus-error@+1 {{invalid instruction}}
qc.lilti x9, x11, -14, 52

# minus-error@+1:1 {{instruction requires the following: 'Xqcicli' (Qualcomm uC Conditional Load Immediate Extension)}}
qc.lilti x9, x11, -14, 2


# plus-error@+2:10 {{register must be a GPR excluding zero (x0)}}
# minus-error@+1:1 {{invalid instruction}}
qc.linei x0, x1, 10, 12

# plus-error@+2:14 {{register must be a GPR excluding zero (x0)}}
# minus-error@+1:1 {{invalid instruction}}
qc.linei x5, x0, 10, 12

# plus-error@+2:20 {{too few operands for instruction}}
# minus-error@+1:1 {{invalid instruction}}
qc.linei x5, x1, 10

# plus-error@+2:18 {{immediate must be an integer in the range [-16, 15]}}
# minus-error@+1 {{invalid instruction}}
qc.linei x5, x1, 130, 12

# plus-error@+2:22 {{immediate must be an integer in the range [-16, 15]}}
# minus-error@+1 {{invalid instruction}}
qc.linei x5, x1, 10, 124

# minus-error@+1:1 {{instruction requires the following: 'Xqcicli' (Qualcomm uC Conditional Load Immediate Extension)}}
qc.linei x5, x1, 10, 12


# plus-error@+2:11 {{register must be a GPR excluding zero (x0)}}
# minus-error@+1:1 {{invalid instruction}}
qc.ligeui x0, x12, 7, -12

# plus-error@+2:15 {{register must be a GPR excluding zero (x0)}}
# minus-error@+1:1 {{invalid instruction}}
qc.ligeui x2, x0, 7, -12

# plus-error@+2:21 {{too few operands for instruction}}
# minus-error@+1:1 {{invalid instruction}}
qc.ligeui x2, x12, 7

# plus-error@+2:20 {{immediate must be an integer in the range [0, 31]}}
# minus-error@+1 {{invalid instruction}}
qc.ligeui x2, x12, -7, -12

# plus-error@+2:23 {{immediate must be an integer in the range [-16, 15]}}
# minus-error@+1 {{invalid instruction}}
qc.ligeui x2, x12, 7, -17

# minus-error@+1:1 {{instruction requires the following: 'Xqcicli' (Qualcomm uC Conditional Load Immediate Extension)}}
qc.ligeui x2, x12, 7, -12


# plus-error@+2:11 {{register must be a GPR excluding zero (x0)}}
# minus-error@+1:1 {{invalid instruction}}
qc.liltui x0, x25, 31, 12

# plus-error@+2:15 {{register must be a GPR excluding zero (x0)}}
# minus-error@+1:1 {{invalid instruction}}
qc.liltui x3, x0, 31, 12

# plus-error@+2:22 {{too few operands for instruction}}
# minus-error@+1:1 {{invalid instruction}}
qc.liltui x3, x25, 31

# plus-error@+2:20 {{immediate must be an integer in the range [0, 31]}}
# minus-error@+1 {{invalid instruction}}
qc.liltui x3, x25, 32, 12

# plus-error@+2:24 {{immediate must be an integer in the range [-16, 15]}}
# minus-error@+1 {{invalid instruction}}
qc.liltui x3, x25, 31, 112

# minus-error@+1:1 {{instruction requires the following: 'Xqcicli' (Qualcomm uC Conditional Load Immediate Extension)}}
qc.liltui x3, x25, 31, 12
