# Xqcicsr - Qualcomm uC CSR Extension
# RUN: llvm-mc --verify=plus -triple riscv32 -mattr=+xqcicsr < %s
# RUN: llvm-mc --verify=minus -triple riscv32 -mattr=-xqcicsr < %s

qc.csrrwr x10, x5, x0
# plus-error@-1:20 {{register must be a GPR excluding zero (x0)}}
# minus-error@-2:1 {{invalid instruction}}

qc.csrrwr x10, x5
# plus-error@-1:18 {{too few operands for instruction}}
# minus-error@-2:1 {{invalid instruction}}

qc.csrrwr x10, x5, x20 # minus-error:1 {{instruction requires the following: 'Xqcicsr' (Qualcomm uC CSR Extension)}}


qc.csrrwri x20, 31, x0
# plus-error@-1:21 {{register must be a GPR excluding zero (x0)}}
# minus-error@-2:1 {{invalid instruction}}

qc.csrrwri x20, 45, x12
# plus-error@-1:17 {{immediate must be an integer in the range [0, 31]}}
# minus-error@-2 {{invalid instruction}}

qc.csrrwri x20, 23
# plus-error@-1:19 {{too few operands for instruction}}
# minus-error@-2:1 {{invalid instruction}}

qc.csrrwri x30, 31, x12 # minus-error:1 {{instruction requires the following: 'Xqcicsr' (Qualcomm uC CSR Extension)}}
