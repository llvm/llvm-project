# Xqcisync - Qualcomm uC Sync Delay Extension
# RUN: llvm-mc --verify=plus -triple riscv32 -mattr=+xqcisync < %s
# RUN: llvm-mc --verify=minus -triple riscv32 -mattr=-xqcisync < %s

qc.c.delay 34
# plus-error@-1:12 {{immediate must be an integer in the range [1, 31]}}
# minus-error@-2 {{invalid instruction}}

qc.c.delay 11, 12
# plus-error@-1:16 {{unexpected extra operand for instruction}}
# minus-error@-2:1 {{invalid instruction}}

qc.c.delay
# plus-error@-1:1 {{too few operands for instruction}}
# minus-error@-2:1 {{invalid instruction}}

qc.c.delay   10 # minus-error:1 {{instruction requires the following: 'Xqcisync' (Qualcomm uC Sync Delay Extension)}}


qc.sync 45
# plus-error@-1:9 {{immediate must be an integer in the range [0, 31]}}
# minus-error@-2 {{invalid instruction}}

qc.sync 22, x4
# plus-error@-1:13 {{unexpected extra operand for instruction}}
# minus-error@-2:1 {{invalid instruction}}

qc.sync
# plus-error@-1:1 {{too few operands for instruction}}
# minus-error@-2:1 {{invalid instruction}}

qc.sync 8 # minus-error:1 {{instruction requires the following: 'Xqcisync' (Qualcomm uC Sync Delay Extension)}}


qc.syncr 56
# plus-error@-1:10 {{immediate must be an integer in the range [0, 31]}}
# minus-error@-2 {{invalid instruction}}

qc.syncr 31, 45
# plus-error@-1:14 {{unexpected extra operand for instruction}}
# minus-error@-2:1 {{invalid instruction}}

qc.syncr
# plus-error@-1:1 {{too few operands for instruction}}
# minus-error@-2:1 {{invalid instruction}}

qc.syncr   23 # minus-error:1 {{instruction requires the following: 'Xqcisync' (Qualcomm uC Sync Delay Extension)}}


qc.syncwf 88
# plus-error@-1:11 {{immediate must be an integer in the range [0, 31]}}
# minus-error@-2 {{invalid instruction}}

qc.syncwf 5, 44
# plus-error@-1:14 {{unexpected extra operand for instruction}}
# minus-error@-2:1 {{invalid instruction}}

qc.syncwf
# plus-error@-1:1 {{too few operands for instruction}}
# minus-error@-2:1 {{invalid instruction}}

qc.syncwf  31 # minus-error:1 {{instruction requires the following: 'Xqcisync' (Qualcomm uC Sync Delay Extension)}}


qc.syncwl 99
# plus-error@-1:11 {{immediate must be an integer in the range [0, 31]}}
# minus-error@-2 {{invalid instruction}}

qc.syncwl 11, x10
# plus-error@-1:15 {{unexpected extra operand for instruction}}
# minus-error@-2:1 {{invalid instruction}}

qc.syncwl
# plus-error@-1:1 {{too few operands for instruction}}
# minus-error@-2:1 {{invalid instruction}}

qc.syncwl  1 # minus-error:1 {{instruction requires the following: 'Xqcisync' (Qualcomm uC Sync Delay Extension)}}


qc.c.sync 45
# plus-error@-1:11 {{immediate must be one of: 0, 1, 2, 4, 8, 15, 16, 31}}
# minus-error@-2 {{invalid instruction}}

qc.c.sync 31, x4
# plus-error@-1:15 {{unexpected extra operand for instruction}}
# minus-error@-2:1 {{invalid instruction}}

qc.c.sync
# plus-error@-1:1 {{too few operands for instruction}}
# minus-error@-2:1 {{invalid instruction}}

qc.c.sync 8 # minus-error:1 {{instruction requires the following: 'Xqcisync' (Qualcomm uC Sync Delay Extension)}}


qc.c.syncr 56
# plus-error@-1:12 {{immediate must be one of: 0, 1, 2, 4, 8, 15, 16, 31}}
# minus-error@-2 {{invalid instruction}}

qc.c.syncr 31, 45
# plus-error@-1:16 {{unexpected extra operand for instruction}}
# minus-error@-2:1 {{invalid instruction}}

qc.c.syncr
# plus-error@-1:1 {{too few operands for instruction}}
# minus-error@-2:1 {{invalid instruction}}

qc.c.syncr   8 # minus-error:1 {{instruction requires the following: 'Xqcisync' (Qualcomm uC Sync Delay Extension)}}


qc.c.syncwf 88
# plus-error@-1:13 {{immediate must be one of: 0, 1, 2, 4, 8, 15, 16, 31}}
# minus-error@-2 {{invalid instruction}}

qc.c.syncwf 8, 44
# plus-error@-1:16 {{unexpected extra operand for instruction}}
# minus-error@-2:1 {{invalid instruction}}

qc.c.syncwf
# plus-error@-1:1 {{too few operands for instruction}}
# minus-error@-2:1 {{invalid instruction}}

qc.c.syncwf  31 # minus-error:1 {{instruction requires the following: 'Xqcisync' (Qualcomm uC Sync Delay Extension)}}


qc.c.syncwl 99
# plus-error@-1:13 {{immediate must be one of: 0, 1, 2, 4, 8, 15, 16, 31}}
# minus-error@-2 {{invalid instruction}}

qc.c.syncwl 15, x10
# plus-error@-1:17 {{unexpected extra operand for instruction}}
# minus-error@-2:1 {{invalid instruction}}

qc.c.syncwl
# plus-error@-1:1 {{too few operands for instruction}}
# minus-error@-2:1 {{invalid instruction}}

qc.c.syncwl  1 # minus-error:1 {{instruction requires the following: 'Xqcisync' (Qualcomm uC Sync Delay Extension)}}
