# Xqcilb - Qualcomm uC Long Branch Extension
# RUN: llvm-mc --verify=expected,plus -triple riscv32 -mattr=+xqcilb < %s
# RUN: llvm-mc --verify=expected,minus -triple riscv32 -mattr=-xqcilb < %s

qc.e.j
# plus-error@-1:1 {{too few operands for instruction}}
# minus-error@-2:1 {{invalid instruction}}

qc.e.j  -2147483649
# plus-error@-1:9 {{operand must be a multiple of 2 bytes in the range [-2147483648, 2147483646]}}
# minus-error@-2 {{invalid instruction}}

qc.e.j  -2147483648 # minus-error:1 {{instruction requires the following: 'Xqcilb' (Qualcomm uC Long Branch Extension)}}

qc.e.j  foo # minus-error:1 {{instruction requires the following: 'Xqcilb' (Qualcomm uC Long Branch Extension)}}

qc.e.jal
# plus-error@-1:1 {{too few operands for instruction}}
# minus-error@-2:1 {{invalid instruction}}

qc.e.jal 2147483649
# plus-error@-1:10 {{operand must be a multiple of 2 bytes in the range [-2147483648, 2147483646]}}
# minus-error@-2 {{invalid instruction}}

qc.e.jal 2147483640 # minus-error:1 {{instruction requires the following: 'Xqcilb' (Qualcomm uC Long Branch Extension)}}

qc.e.j foo@plt # expected-error:11 {{unexpected token}}
