# Xqcisls - Qualcomm uC Scaled Load Store Extension
# RUN: llvm-mc --verify=plus -triple riscv32 -mattr=+xqcisls < %s
# RUN: llvm-mc --verify=minus -triple riscv32 -mattr=-xqcisls < %s

qc.lrb x5, x2, x0, 4
# plus-error@-1:16 {{register must be a GPR excluding zero (x0)}}
# minus-error@-2:1 {{invalid instruction}}

qc.lrb x5, x2, x4
# plus-error@-1:18 {{too few operands for instruction}}
# minus-error@-2:1 {{invalid instruction}}

qc.lrb x5, x2, x4, 12
# plus-error@-1:20 {{immediate must be an integer in the range [0, 7]}}
# minus-error@-2:1 {{invalid instruction}}

qc.lrb x5, 2, x4, 4
# plus-error@-1:12 {{register must be a GPR}}
# minus-error@-2:1 {{invalid instruction}}

qc.lrb x5, x2, x4, 4 # minus-error:1 {{instruction requires the following: 'Xqcisls' (Qualcomm uC Scaled Load Store Extension)}}


qc.lrh x1, x12, x0, 2
# plus-error@-1:17 {{register must be a GPR excluding zero (x0)}}
# minus-error@-2:1 {{invalid instruction}}

qc.lrh x1, x12, x6
# plus-error@-1:19 {{too few operands for instruction}}
# minus-error@-2:1 {{invalid instruction}}

qc.lrh x1, x12, x6, 22
# plus-error@-1:21 {{immediate must be an integer in the range [0, 7]}}
# minus-error@-2:1 {{invalid instruction}}

qc.lrh x1, 12, x6, 2
# plus-error@-1:12 {{register must be a GPR}}
# minus-error@-2:1 {{invalid instruction}}

qc.lrh x1, x12, x6, 2 # minus-error:1 {{instruction requires the following: 'Xqcisls' (Qualcomm uC Scaled Load Store Extension)}}


qc.lrw x15, x7, x0, 1
# plus-error@-1:17 {{register must be a GPR excluding zero (x0)}}
# minus-error@-2:1 {{invalid instruction}}

qc.lrw x15, x7, x14
# plus-error@-1:20 {{too few operands for instruction}}
# minus-error@-2:1 {{invalid instruction}}

qc.lrw x15, x7, x14, 11
# plus-error@-1:22 {{immediate must be an integer in the range [0, 7]}}
# minus-error@-2:1 {{invalid instruction}}

qc.lrw x15, 7, x14, 1
# plus-error@-1:13 {{register must be a GPR}}
# minus-error@-2:1 {{invalid instruction}}

qc.lrw x15, x7, x14, 1 # minus-error:1 {{instruction requires the following: 'Xqcisls' (Qualcomm uC Scaled Load Store Extension)}}


qc.lrbu x9, x11, x0, 7
# plus-error@-1:18 {{register must be a GPR excluding zero (x0)}}
# minus-error@-2:1 {{invalid instruction}}

qc.lrbu x9, x11, x4
# plus-error@-1:20 {{too few operands for instruction}}
# minus-error@-2:1 {{invalid instruction}}

qc.lrbu x9, x11, x4, 37
# plus-error@-1:22 {{immediate must be an integer in the range [0, 7]}}
# minus-error@-2:1 {{invalid instruction}}

qc.lrbu x9, 11, x4, 7
# plus-error@-1:13 {{register must be a GPR}}
# minus-error@-2:1 {{invalid instruction}}

qc.lrbu x9, x11, x4, 7 # minus-error:1 {{instruction requires the following: 'Xqcisls' (Qualcomm uC Scaled Load Store Extension)}}


qc.lrhu x16, x6, x0, 4
# plus-error@-1:18 {{register must be a GPR excluding zero (x0)}}
# minus-error@-2:1 {{invalid instruction}}

qc.lrhu x16, x6, x10
# plus-error@-1:21 {{too few operands for instruction}}
# minus-error@-2:1 {{invalid instruction}}

qc.lrhu x16, x6, x10, 44
# plus-error@-1:23 {{immediate must be an integer in the range [0, 7]}}
# minus-error@-2:1 {{invalid instruction}}

qc.lrhu x16, 6, x10, 4
# plus-error@-1:14 {{register must be a GPR}}
# minus-error@-2:1 {{invalid instruction}}

qc.lrhu x16, x6, x10, 4 # minus-error:1 {{instruction requires the following: 'Xqcisls' (Qualcomm uC Scaled Load Store Extension)}}


qc.srb x0, x2, x0, 3
# plus-error@-1:16 {{register must be a GPR excluding zero (x0)}}
# minus-error@-2:1 {{invalid instruction}}

qc.srb x0, x2, x8
# plus-error@-1:18 {{too few operands for instruction}}
# minus-error@-2:1 {{invalid instruction}}

qc.srb x0, x2, x8, 93
# plus-error@-1:20 {{immediate must be an integer in the range [0, 7]}}
# minus-error@-2:1 {{invalid instruction}}

qc.srb x0, 2, x8, 3
# plus-error@-1:12 {{register must be a GPR}}
# minus-error@-2:1 {{invalid instruction}}

qc.srb x0, x2, x8, 3 # minus-error:1 {{instruction requires the following: 'Xqcisls' (Qualcomm uC Scaled Load Store Extension)}}


qc.srh x13, x0, x0, 6
# plus-error@-1:17 {{register must be a GPR excluding zero (x0)}}
# minus-error@-2:1 {{invalid instruction}}

qc.srh x13, x0, x20
# plus-error@-1:20 {{too few operands for instruction}}
# minus-error@-2:1 {{invalid instruction}}

qc.srh x13, x0, x20, 76
# plus-error@-1:22 {{immediate must be an integer in the range [0, 7]}}
# minus-error@-2:1 {{invalid instruction}}

qc.srh x13, 0, x20, 6
# plus-error@-1:13 {{register must be a GPR}}
# minus-error@-2:1 {{invalid instruction}}

qc.srh x13, x0, x20, 6 # minus-error:1 {{instruction requires the following: 'Xqcisls' (Qualcomm uC Scaled Load Store Extension)}}


qc.srw x17, x18, x0, 0
# plus-error@-1:18 {{register must be a GPR excluding zero (x0)}}
# minus-error@-2:1 {{invalid instruction}}

qc.srw x17, x18, x19
# plus-error@-1:21 {{too few operands for instruction}}
# minus-error@-2:1 {{invalid instruction}}

qc.srw x17, x18, x19, 10
# plus-error@-1:23 {{immediate must be an integer in the range [0, 7]}}
# minus-error@-2:1 {{invalid instruction}}

qc.srw x17, 18, x19, 0
# plus-error@-1:13 {{register must be a GPR}}
# minus-error@-2:1 {{invalid instruction}}

qc.srw x17, x18, x19, 0 # minus-error:1 {{instruction requires the following: 'Xqcisls' (Qualcomm uC Scaled Load Store Extension)}}
