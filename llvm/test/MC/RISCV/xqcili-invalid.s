# Xqcili - Qualcomm uC Load Large Immediate Extension
# RUN: llvm-mc --verify=plus -triple riscv32 -mattr=+xqcili < %s
# RUN: llvm-mc --verify=minus -triple riscv32 -mattr=-xqcili < %s

qc.e.li 9, 33554432
# plus-error@-1:9 {{register must be a GPR excluding zero (x0)}}
# minus-error@-2:1 {{instruction requires the following: 'Xqcili' (Qualcomm uC Load Large Immediate Extension)}}

qc.e.li x9
# plus-error@-1:11 {{too few operands for instruction}}
# minus-error@-2:1 {{invalid instruction}}

qc.e.li x9, 4294967296
# plus-error@-1:1 {{invalid instruction, any one of the following would fix this:}}
# plus-note@-2:13 {{immediate must be an integer in the range [-2147483648, 4294967295]}}
# plus-note@-3:13 {{immediate must be an integer in the range [-2147483648, 4294967295]}}
# minus-error@-4 {{invalid instruction}}

qc.e.li x9, 4294967295 # minus-error:1 {{instruction requires the following: 'Xqcili' (Qualcomm uC Load Large Immediate Extension)}}


qc.li x0, 114514
# plus-error@-1:7 {{register must be a GPR excluding zero (x0)}}
# minus-error@-2:1 {{invalid instruction}}

qc.li x10
# plus-error@-1:10 {{too few operands for instruction}}
# minus-error@-2:1 {{invalid instruction}}

qc.li x10, 33554432
# plus-error@-1:12 {{operand must be a symbol with a %qc.abs20 specifier or an integer in the range [-524288, 524287]}}
# minus-error@-2 {{invalid instruction}}

qc.li x10, 114514 # minus-error:1 {{instruction requires the following: 'Xqcili' (Qualcomm uC Load Large Immediate Extension)}}
