# Xqcisync - Qualcomm uC Sync Delay Extension
# RUN: llvm-mc --verify=plus -triple riscv32 -mattr=+xqcisync < %s
# RUN: llvm-mc --verify=minus -triple riscv32 -mattr=-xqcisync < %s

# plus-error@+2:12 {{immediate must be an integer in the range [1, 31]}}
# minus-error@+1 {{invalid instruction}}
qc.c.delay 34

# plus-error@+2:16 {{unexpected extra operand for instruction}}
# minus-error@+1:1 {{invalid instruction}}
qc.c.delay 11, 12

# plus-error@+2:1 {{too few operands for instruction}}
# minus-error@+1:1 {{invalid instruction}}
qc.c.delay

# minus-error@+1:1 {{instruction requires the following: 'Xqcisync' (Qualcomm uC Sync Delay Extension)}}
qc.c.delay   10


# plus-error@+2:9 {{immediate must be an integer in the range [0, 31]}}
# minus-error@+1 {{invalid instruction}}
qc.sync 45

# plus-error@+2:13 {{unexpected extra operand for instruction}}
# minus-error@+1:1 {{invalid instruction}}
qc.sync 22, x4

# plus-error@+2:1 {{too few operands for instruction}}
# minus-error@+1:1 {{invalid instruction}}
qc.sync

# minus-error@+1:1 {{instruction requires the following: 'Xqcisync' (Qualcomm uC Sync Delay Extension)}}
qc.sync 8


# plus-error@+2:10 {{immediate must be an integer in the range [0, 31]}}
# minus-error@+1 {{invalid instruction}}
qc.syncr 56

# plus-error@+2:14 {{unexpected extra operand for instruction}}
# minus-error@+1:1 {{invalid instruction}}
qc.syncr 31, 45

# plus-error@+2:1 {{too few operands for instruction}}
# minus-error@+1:1 {{invalid instruction}}
qc.syncr

# minus-error@+1:1 {{instruction requires the following: 'Xqcisync' (Qualcomm uC Sync Delay Extension)}}
qc.syncr   23


# plus-error@+2:11 {{immediate must be an integer in the range [0, 31]}}
# minus-error@+1 {{invalid instruction}}
qc.syncwf 88

# plus-error@+2:14 {{unexpected extra operand for instruction}}
# minus-error@+1:1 {{invalid instruction}}
qc.syncwf 5, 44

# plus-error@+2:1 {{too few operands for instruction}}
# minus-error@+1:1 {{invalid instruction}}
qc.syncwf

# minus-error@+1:1 {{instruction requires the following: 'Xqcisync' (Qualcomm uC Sync Delay Extension)}}
qc.syncwf  31


# plus-error@+2:11 {{immediate must be an integer in the range [0, 31]}}
# minus-error@+1 {{invalid instruction}}
qc.syncwl 99

# plus-error@+2:15 {{unexpected extra operand for instruction}}
# minus-error@+1:1 {{invalid instruction}}
qc.syncwl 11, x10

# plus-error@+2:1 {{too few operands for instruction}}
# minus-error@+1:1 {{invalid instruction}}
qc.syncwl

# minus-error@+1:1 {{instruction requires the following: 'Xqcisync' (Qualcomm uC Sync Delay Extension)}}
qc.syncwl  1


# plus-error@+2:11 {{immediate must be one of: 0, 1, 2, 4, 8, 15, 16, 31}}
# minus-error@+1 {{invalid instruction}}
qc.c.sync 45

# plus-error@+2:15 {{unexpected extra operand for instruction}}
# minus-error@+1:1 {{invalid instruction}}
qc.c.sync 31, x4

# plus-error@+2:1 {{too few operands for instruction}}
# minus-error@+1:1 {{invalid instruction}}
qc.c.sync

# minus-error@+1:1 {{instruction requires the following: 'Xqcisync' (Qualcomm uC Sync Delay Extension)}}
qc.c.sync 8


# plus-error@+2:12 {{immediate must be one of: 0, 1, 2, 4, 8, 15, 16, 31}}
# minus-error@+1 {{invalid instruction}}
qc.c.syncr 56

# plus-error@+2:16 {{unexpected extra operand for instruction}}
# minus-error@+1:1 {{invalid instruction}}
qc.c.syncr 31, 45

# plus-error@+2:1 {{too few operands for instruction}}
# minus-error@+1:1 {{invalid instruction}}
qc.c.syncr

# minus-error@+1:1 {{instruction requires the following: 'Xqcisync' (Qualcomm uC Sync Delay Extension)}}
qc.c.syncr   8


# plus-error@+2:13 {{immediate must be one of: 0, 1, 2, 4, 8, 15, 16, 31}}
# minus-error@+1 {{invalid instruction}}
qc.c.syncwf 88

# plus-error@+2:16 {{unexpected extra operand for instruction}}
# minus-error@+1:1 {{invalid instruction}}
qc.c.syncwf 8, 44

# plus-error@+2:1 {{too few operands for instruction}}
# minus-error@+1:1 {{invalid instruction}}
qc.c.syncwf

# minus-error@+1:1 {{instruction requires the following: 'Xqcisync' (Qualcomm uC Sync Delay Extension)}}
qc.c.syncwf  31


# plus-error@+2:13 {{immediate must be one of: 0, 1, 2, 4, 8, 15, 16, 31}}
# minus-error@+1 {{invalid instruction}}
qc.c.syncwl 99

# plus-error@+2:17 {{unexpected extra operand for instruction}}
# minus-error@+1:1 {{invalid instruction}}
qc.c.syncwl 15, x10

# plus-error@+2:1 {{too few operands for instruction}}
# minus-error@+1:1 {{invalid instruction}}
qc.c.syncwl

# minus-error@+1:1 {{instruction requires the following: 'Xqcisync' (Qualcomm uC Sync Delay Extension)}}
qc.c.syncwl  1
