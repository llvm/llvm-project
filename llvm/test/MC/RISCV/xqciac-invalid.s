# Xqciac - Qualcomm uC Load-Store Address Calculation Extension
# RUN: llvm-mc --verify=plus -triple riscv32 -mattr=+xqciac < %s
# RUN: llvm-mc --verify=minus -triple riscv32 -mattr=-xqciac < %s

qc.c.muliadd x5, x10, 4
# plus-error@-1:14 {{register must be a GPR from x8 to x15}}
# minus-error@-2:1 {{invalid instruction}}

qc.c.muliadd x15
# plus-error@-1:17 {{too few operands for instruction}}
# minus-error@-2:1 {{invalid instruction}}

qc.c.muliadd x10, x15, 32
# plus-error@-1:24 {{immediate must be an integer in the range [0, 31]}}
# minus-error@-2 {{invalid instruction}}

qc.c.muliadd x10, x15, 20 # minus-error:1 {{instruction requires the following: 'Xqciac' (Qualcomm uC Load-Store Address Calculation Extension)}}


qc.muliadd x0, x10, 1048577
# plus-error@-1:12 {{register must be a GPR excluding zero (x0)}}
# minus-error@-2:1 {{instruction requires the following: 'Xqciac' (Qualcomm uC Load-Store Address Calculation Extension)}}

qc.muliadd x10
# plus-error@-1:15 {{too few operands for instruction}}
# minus-error@-2:1 {{invalid instruction}}

qc.muliadd x10, x15, 8589934592
# plus-error@-1:22 {{operand must be a symbol with %lo/%pcrel_lo/%tprel_lo specifier or an integer in the range [-2048, 2047]}}
# minus-error@-2 {{invalid instruction}}

qc.muliadd x10, x15, 577 # minus-error:1 {{instruction requires the following: 'Xqciac' (Qualcomm uC Load-Store Address Calculation Extension)}}


qc.shladd 0, x10, 1048577
# plus-error@-1:11 {{register must be a GPR excluding zero (x0)}}
# minus-error@-2:1 {{instruction requires the following: 'Xqciac' (Qualcomm uC Load-Store Address Calculation Extension)}}

qc.shladd x10
# plus-error@-1:14 {{too few operands for instruction}}
# minus-error@-2:1 {{invalid instruction}}

qc.shladd x10, x15, x11, 2
# plus-error@-1:26 {{immediate must be an integer in the range [4, 31]}}
# minus-error@-2 {{invalid instruction}}

qc.shladd x10, x15, x11, 5 # minus-error:1 {{instruction requires the following: 'Xqciac' (Qualcomm uC Load-Store Address Calculation Extension)}}
