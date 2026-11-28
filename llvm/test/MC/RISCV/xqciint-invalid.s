# Xqciint - Qualcomm uC Interrupts extension
# RUN: llvm-mc --verify=plus -triple riscv32 -mattr=+xqciint < %s
# RUN: llvm-mc --verify=minus -triple riscv32 -mattr=-xqciint < %s

qc.setinti 1025
# plus-error@-1:12 {{immediate must be an integer in the range [0, 1023]}}
# minus-error@-2:1 {{invalid instruction}}

qc.setinti 11, 12
# plus-error@-1:16 {{unexpected extra operand for instruction}}
# minus-error@-2:1 {{invalid instruction}}

qc.setinti
# plus-error@-1:1 {{too few operands for instruction}}
# minus-error@-2:1 {{invalid instruction}}

qc.setinti   10 # minus-error:1 {{instruction requires the following: 'Xqciint' (Qualcomm uC Interrupts Extension)}}


qc.clrinti 2000
# plus-error@-1:12 {{immediate must be an integer in the range [0, 1023]}}
# minus-error@-2 {{invalid instruction}}

qc.clrinti 22, x4
# plus-error@-1:16 {{unexpected extra operand for instruction}}
# minus-error@-2:1 {{invalid instruction}}

qc.clrinti
# plus-error@-1:1 {{too few operands for instruction}}
# minus-error@-2:1 {{invalid instruction}}

qc.clrinti 8 # minus-error:1 {{instruction requires the following: 'Xqciint' (Qualcomm uC Interrupts Extension)}}


qc.c.clrint 22
# plus-error@-1:13 {{register must be a GPR excluding zero (x0)}}
# minus-error@-2:1 {{invalid instruction}}

qc.c.clrint
# plus-error@-1:1 {{too few operands for instruction}}
# minus-error@-2:1 {{invalid instruction}}

qc.c.clrint x8 # minus-error:1 {{instruction requires the following: 'Xqciint' (Qualcomm uC Interrupts Extension)}}


qc.c.di 22
# plus-error@-1:9 {{unexpected extra operand for instruction}}
# minus-error@-2:1 {{invalid instruction}}

qc.c.di # minus-error:1 {{instruction requires the following: 'Xqciint' (Qualcomm uC Interrupts Extension)}}


qc.c.dir 22
# plus-error@-1:10 {{register must be a GPR excluding zero (x0)}}
# minus-error@-2:1 {{invalid instruction}}

qc.c.dir
# plus-error@-1:1 {{too few operands for instruction}}
# minus-error@-2:1 {{invalid instruction}}

qc.c.dir x8 # minus-error:1 {{instruction requires the following: 'Xqciint' (Qualcomm uC Interrupts Extension)}}


qc.c.ei 22
# plus-error@-1:9 {{unexpected extra operand for instruction}}
# minus-error@-2:1 {{invalid instruction}}

qc.c.ei # minus-error:1 {{instruction requires the following: 'Xqciint' (Qualcomm uC Interrupts Extension)}}


qc.c.eir 22
# plus-error@-1:10 {{register must be a GPR excluding zero (x0)}}
# minus-error@-2:1 {{invalid instruction}}

qc.c.eir
# plus-error@-1:1 {{too few operands for instruction}}
# minus-error@-2:1 {{invalid instruction}}

qc.c.eir x8 # minus-error:1 {{instruction requires the following: 'Xqciint' (Qualcomm uC Interrupts Extension)}}


qc.c.mienter.nest 22
# plus-error@-1:19 {{unexpected extra operand for instruction}}
# minus-error@-2:1 {{invalid instruction}}

qc.c.mienter.nest # minus-error:1 {{instruction requires the following: 'Xqciint' (Qualcomm uC Interrupts Extension)}}


qc.c.mienter 22
# plus-error@-1:14 {{unexpected extra operand for instruction}}
# minus-error@-2:1 {{invalid instruction}}

qc.c.mienter # minus-error:1 {{instruction requires the following: 'Xqciint' (Qualcomm uC Interrupts Extension)}}


qc.c.mileaveret 22
# plus-error@-1:17 {{unexpected extra operand for instruction}}
# minus-error@-2:1 {{invalid instruction}}

qc.c.mileaveret # minus-error:1 {{instruction requires the following: 'Xqciint' (Qualcomm uC Interrupts Extension)}}


qc.c.setint 22
# plus-error@-1:13 {{register must be a GPR excluding zero (x0)}}
# minus-error@-2:1 {{invalid instruction}}

qc.c.setint
# plus-error@-1:1 {{too few operands for instruction}}
# minus-error@-2:1 {{invalid instruction}}

qc.c.setint x8 # minus-error:1 {{instruction requires the following: 'Xqciint' (Qualcomm uC Interrupts Extension)}}


qc.c.mret x8
# plus-error@-1:11 {{unexpected extra operand for instruction}}
# minus-error@-2:1 {{invalid instruction}}

# CHECK-EXT: :[[@LINE+1]]:1: error: instruction requires the following: 'Xqciint' (Qualcomm uC Interrupts Extension)
qc.c.mret # minus-error {{instruction requires the following: 'Xqciint' (Qualcomm uC Interrupts Extension)}}


qc.c.mnret 10
# plus-error@-1:12 {{unexpected extra operand for instruction}}
# minus-error@-2:1 {{invalid instruction}}

# CHECK-EXT: :[[@LINE+1]]:1: error: instruction requires the following: 'Xqciint' (Qualcomm uC Interrupts Extension)
qc.c.mnret # minus-error {{instruction requires the following: 'Xqciint' (Qualcomm uC Interrupts Extension)}}
