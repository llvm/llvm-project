# Xqcicm - Qualcomm uC Conditional Move Extension
# RUN: llvm-mc --verify=plus -triple riscv32 -mattr=+xqcicm < %s
# RUN: llvm-mc --verify=minus -triple riscv32 -mattr=-xqcicm < %s

qc.c.mveqz 9, x10
# plus-error@-1:12 {{register must be a GPR from x8 to x15}}
# minus-error@-2:1 {{invalid instruction}}

qc.c.mveqz x9
# plus-error@-1:14 {{too few operands for instruction}}
# minus-error@-2:1 {{invalid instruction}}

qc.c.mveqz x9, x10 # minus-error:1 {{instruction requires the following: 'Xqcicm' (Qualcomm uC Conditional Move Extension)}}


qc.mveq 9, x10, x11, x12
# plus-error@-1:9 {{register must be a GPR excluding zero (x0)}}
# minus-error@-2:1 {{invalid instruction}}

qc.mveq x9
# plus-error@-1:11 {{too few operands for instruction}}
# minus-error@-2:1 {{invalid instruction}}

qc.mveq x9, x10, x11, x12 # minus-error:1 {{instruction requires the following: 'Xqcicm' (Qualcomm uC Conditional Move Extension)}}


qc.mvge 9, x10, x11, x12
# plus-error@-1:9 {{register must be a GPR excluding zero (x0)}}
# minus-error@-2:1 {{invalid instruction}}

qc.mvge x9
# plus-error@-1:11 {{too few operands for instruction}}
# minus-error@-2:1 {{invalid instruction}}

qc.mvge x9, x10, x11, x12 # minus-error:1 {{instruction requires the following: 'Xqcicm' (Qualcomm uC Conditional Move Extension)}}


qc.mvgeu 9, x10, x11, x12
# plus-error@-1:10 {{register must be a GPR excluding zero (x0)}}
# minus-error@-2:1 {{invalid instruction}}

qc.mvgeu x9
# plus-error@-1:12 {{too few operands for instruction}}
# minus-error@-2:1 {{invalid instruction}}

qc.mvgeu x9, x10, x11, x12 # minus-error:1 {{instruction requires the following: 'Xqcicm' (Qualcomm uC Conditional Move Extension)}}


qc.mvlt 9, x10, x11, x12
# plus-error@-1:9 {{register must be a GPR excluding zero (x0)}}
# minus-error@-2:1 {{invalid instruction}}

qc.mvlt x9
# plus-error@-1:11 {{too few operands for instruction}}
# minus-error@-2:1 {{invalid instruction}}

qc.mvlt x9, x10, x11, x12 # minus-error:1 {{instruction requires the following: 'Xqcicm' (Qualcomm uC Conditional Move Extension)}}


qc.mvltu 9, x10, x11, x12
# plus-error@-1:10 {{register must be a GPR excluding zero (x0)}}
# minus-error@-2:1 {{invalid instruction}}

qc.mvltu x9
# plus-error@-1:12 {{too few operands for instruction}}
# minus-error@-2:1 {{invalid instruction}}

qc.mvltu x9, x10, x11, x12 # minus-error:1 {{instruction requires the following: 'Xqcicm' (Qualcomm uC Conditional Move Extension)}}


qc.mvne 9, x10, x11, x12
# plus-error@-1:9 {{register must be a GPR excluding zero (x0)}}
# minus-error@-2:1 {{invalid instruction}}

qc.mvne x9
# plus-error@-1:11 {{too few operands for instruction}}
# minus-error@-2:1 {{invalid instruction}}

qc.mvne x9, x10, x11, x12 # minus-error:1 {{instruction requires the following: 'Xqcicm' (Qualcomm uC Conditional Move Extension)}}


qc.mveqi 9, x10, 5, x12
# plus-error@-1:10 {{register must be a GPR excluding zero (x0)}}
# minus-error@-2:1 {{invalid instruction}}

qc.mveqi x9
# plus-error@-1:12 {{too few operands for instruction}}
# minus-error@-2:1 {{invalid instruction}}

qc.mveqi x9, x10, 17, x12
# plus-error@-1:19 {{immediate must be an integer in the range [-16, 15]}}
# minus-error@-2 {{invalid instruction}}

qc.mveqi x9, x10, 5, x12 # minus-error:1 {{instruction requires the following: 'Xqcicm' (Qualcomm uC Conditional Move Extension)}}


qc.mvgei 9, x10, 5, x12
# plus-error@-1:10 {{register must be a GPR excluding zero (x0)}}
# minus-error@-2:1 {{invalid instruction}}

qc.mvgei x9
# plus-error@-1:12 {{too few operands for instruction}}
# minus-error@-2:1 {{invalid instruction}}

qc.mvgei x9, x10, 17, x12
# plus-error@-1:19 {{immediate must be an integer in the range [-16, 15]}}
# minus-error@-2 {{invalid instruction}}

qc.mvgei x9, x10, 5, x12 # minus-error:1 {{instruction requires the following: 'Xqcicm' (Qualcomm uC Conditional Move Extension)}}


qc.mvlti 9, x10, 5, x12
# plus-error@-1:10 {{register must be a GPR excluding zero (x0)}}
# minus-error@-2:1 {{invalid instruction}}

qc.mvlti x9
# plus-error@-1:12 {{too few operands for instruction}}
# minus-error@-2:1 {{invalid instruction}}

qc.mvlti x9, x10, 17, x12
# plus-error@-1:19 {{immediate must be an integer in the range [-16, 15]}}
# minus-error@-2 {{invalid instruction}}

qc.mvlti x9, x10, 5, x12 # minus-error:1 {{instruction requires the following: 'Xqcicm' (Qualcomm uC Conditional Move Extension)}}


qc.mvnei 9, x10, 5, x12
# plus-error@-1:10 {{register must be a GPR excluding zero (x0)}}
# minus-error@-2:1 {{invalid instruction}}

qc.mvnei x9
# plus-error@-1:12 {{too few operands for instruction}}
# minus-error@-2:1 {{invalid instruction}}

qc.mvnei x9, x10, 17, x12
# plus-error@-1:19 {{immediate must be an integer in the range [-16, 15]}}
# minus-error@-2 {{invalid instruction}}

qc.mvnei x9, x10, 5, x12 # minus-error:1 {{instruction requires the following: 'Xqcicm' (Qualcomm uC Conditional Move Extension)}}


qc.mvltui 9, x10, 5, x12
# plus-error@-1:11 {{register must be a GPR excluding zero (x0)}}
# minus-error@-2:1 {{invalid instruction}}

qc.mvltui x9
# plus-error@-1:13 {{too few operands for instruction}}
# minus-error@-2:1 {{invalid instruction}}

qc.mvltui x9, x10, 37, x12
# plus-error@-1:20 {{immediate must be an integer in the range [0, 31]}}
# minus-error@-2 {{invalid instruction}}

qc.mvltui x9, x10, 5, x12 # minus-error:1 {{instruction requires the following: 'Xqcicm' (Qualcomm uC Conditional Move Extension)}}


qc.mvgeui 9, x10, 5, x12
# plus-error@-1:11 {{register must be a GPR excluding zero (x0)}}
# minus-error@-2:1 {{invalid instruction}}

qc.mvgeui x9
# plus-error@-1:13 {{too few operands for instruction}}
# minus-error@-2:1 {{invalid instruction}}

qc.mvgeui x9, x10, 37, x12
# plus-error@-1:20 {{immediate must be an integer in the range [0, 31]}}
# minus-error@-2 {{invalid instruction}}

qc.mvgeui x9, x10, 5, x12 # minus-error:1 {{instruction requires the following: 'Xqcicm' (Qualcomm uC Conditional Move Extension)}}
