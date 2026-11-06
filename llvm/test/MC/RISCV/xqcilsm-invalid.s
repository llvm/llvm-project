# Xqcilsm - Qualcomm uC Load Store Multiple Extension
# RUN: llvm-mc --verify=expected,plus -triple riscv32 -mattr=+xqcilsm < %s
# RUN: llvm-mc --verify=expected,minus -triple riscv32 -mattr=-xqcilsm < %s

# expected-error@+1:20 {{expected register}}
qc.swm x5, x20, 12(20)

# plus-error@+2:8 {{register must be a GPR excluding zero (x0)}}
# minus-error@+1:1 {{invalid instruction}}
qc.swm x0, x20, 12(x3)

# plus-error@+2:12 {{register must be a GPR excluding zero (x0)}}
# minus-error@+1:1 {{invalid instruction}}
qc.swm x5, x0, 12(x3)

# plus-error@+2:14 {{too few operands for instruction}}
# minus-error@+1:1 {{invalid instruction}}
qc.swm x5, x3

# plus-error@+2:17 {{immediate must be a multiple of 4 bytes in the range [0, 124]}}
# minus-error@+1:1 {{invalid instruction}}
qc.swm x5, x20, 45(x3)

# minus-error@+1:1 {{instruction requires the following: 'Xqcilsm' (Qualcomm uC Load Store Multiple Extension)}}
qc.swm x5, x20, 12(x3)

# expected-error@+1:20 {{expected register}}
qc.swmi x10, 4, 20(4)

# plus-error@+2:9 {{register must be a GPR excluding zero (x0)}}
# minus-error@+1:1 {{invalid instruction}}
qc.swmi x0, 4, 20(x4)

# plus-error@+2:19 {{too few operands for instruction}}
# minus-error@+1:1 {{invalid instruction}}
qc.swmi x10, 4, 20

# plus-error@+2:14 {{immediate must be an integer in the range [1, 31]}}
# minus-error@+1:1 {{invalid instruction}}
qc.swmi x10, 32, 20(x4)

# plus-error@+2:14 {{immediate must be an integer in the range [1, 31]}}
# minus-error@+1:1 {{invalid instruction}}
qc.swmi x10, 0, 20(x4)

# plus-error@+2:17 {{immediate must be a multiple of 4 bytes in the range [0, 124]}}
# minus-error@+1:1 {{invalid instruction}}
qc.swmi x10, 4, 45(x4)

# minus-error@+1:1 {{instruction requires the following: 'Xqcilsm' (Qualcomm uC Load Store Multiple Extension)}}
qc.swmi x10, 4, 20(x4)

# expected-error@+1:23 {{expected register}}
qc.setwm x4, x30, 124(2)

# plus-error@+2:14 {{register must be a GPR excluding zero (x0)}}
# minus-error@+1:1 {{invalid instruction}}
qc.setwm x4, x0, 124(x2)

# plus-error@+2:22 {{too few operands for instruction}}
# minus-error@+1:1 {{invalid instruction}}
qc.setwm x4, x30, 124

# plus-error@+2:19 {{immediate must be a multiple of 4 bytes in the range [0, 124]}}
# minus-error@+1:1 {{invalid instruction}}
qc.setwm x4, x30, 128(x2)

# minus-error@+1:1 {{instruction requires the following: 'Xqcilsm' (Qualcomm uC Load Store Multiple Extension)}}
qc.setwm x4, x30, 124(x2)

# expected-error@+1:22 {{expected register}}
qc.setwmi x5, 31, 12(12)

# plus-error@+2:21 {{too few operands for instruction}}
# minus-error@+1:1 {{invalid instruction}}
qc.setwmi x5, 31, 12

# plus-error@+2:15 {{immediate must be an integer in the range [1, 31]}}
# minus-error@+1:1 {{invalid instruction}}
qc.setwmi x5, 37, 12(x12)

# plus-error@+2:15 {{immediate must be an integer in the range [1, 31]}}
# minus-error@+1:1 {{invalid instruction}}
qc.setwmi x5, 0, 12(x12)

# plus-error@+2:19 {{immediate must be a multiple of 4 bytes in the range [0, 124]}}
# minus-error@+1:1 {{invalid instruction}}
qc.setwmi x5, 31, 98(x12)

# minus-error@+1:1 {{instruction requires the following: 'Xqcilsm' (Qualcomm uC Load Store Multiple Extension)}}
qc.setwmi x5, 31, 12(x12)

# expected-error@+1:19 {{expected register}}
qc.lwm x7, x1, 24(20)

# plus-error@+2:18 {{too few operands for instruction}}
# minus-error@+1:1 {{invalid instruction}}
qc.lwm x7, x1, 24

# plus-error@+2:8 {{register must be a GPR excluding zero (x0)}}
# minus-error@+1:1 {{invalid instruction}}
qc.lwm x0, x1, 24(x20)

# plus-error@+2:16 {{immediate must be a multiple of 4 bytes in the range [0, 124]}}
# minus-error@+1:1 {{invalid instruction}}
qc.lwm x7, x1, 46(x20)

# minus-error@+1:1 {{instruction requires the following: 'Xqcilsm' (Qualcomm uC Load Store Multiple Extension)}}
qc.lwm x7, x1, 24(x20)

# expected-error@+1:19 {{expected register}}
qc.lwmi x13, 9, 4(23)

# plus-error@+2:18 {{too few operands for instruction}}
# minus-error@+1:1 {{invalid instruction}}
qc.lwmi x13, 9, 4

# plus-error@+2:9 {{register must be a GPR excluding zero (x0)}}
# minus-error@+1:1 {{invalid instruction}}
qc.lwmi x0, 9, 4(x23)

# plus-error@+2:14 {{immediate must be an integer in the range [1, 31]}}
# minus-error@+1:1 {{invalid instruction}}
qc.lwmi x13, 44, 4(x23)

# plus-error@+2:14 {{immediate must be an integer in the range [1, 31]}}
# minus-error@+1:1 {{invalid instruction}}
qc.lwmi x13, 0, 4(x23)

# plus-error@+2:17 {{immediate must be a multiple of 4 bytes in the range [0, 124]}}
# minus-error@+1:1 {{invalid instruction}}
qc.lwmi x13, 9, 77(x23)

# minus-error@+1:1 {{instruction requires the following: 'Xqcilsm' (Qualcomm uC Load Store Multiple Extension)}}
qc.lwmi x13, 9, 4(x23)
