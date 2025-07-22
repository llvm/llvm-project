# Xqcisync - Qualcomm uC Sync Delay Extension
# RUN: llvm-mc %s -triple=riscv32 -mattr=+experimental-xqcisync -M no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ENC,CHECK-INST,CHECK-NOALIAS %s
# RUN: llvm-mc -filetype=obj -triple riscv32 -mattr=+experimental-xqcisync < %s \
# RUN:     | llvm-objdump --mattr=+experimental-xqcisync -M no-aliases --no-print-imm-hex -d - \
# RUN:     | FileCheck -check-prefix=CHECK-INST %s
# RUN: llvm-mc %s -triple=riscv32 -mattr=+experimental-xqcisync -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ENC,CHECK-INST,CHECK-ALIAS %s
# RUN: llvm-mc -filetype=obj -triple riscv32 -mattr=+experimental-xqcisync < %s \
# RUN:     | llvm-objdump --mattr=+experimental-xqcisync --no-print-imm-hex -d - \
# RUN:     | FileCheck -check-prefix=CHECK-INST %s

# CHECK-INST: qc.c.delay     10
# CHECK-ENC: encoding: [0x2a,0x00]
qc.c.delay 10

# CHECK-INST: qc.sync      9
# CHECK-ENC: encoding: [0x13,0x30,0x90,0x10]
qc.sync 9

# CHECK-INST: qc.syncr     23
# CHECK-ENC: encoding: [0x13,0x30,0x70,0x21]
qc.syncr 23

# CHECK-INST: qc.syncwf    30
# CHECK-ENC: encoding: [0x13,0x30,0xe0,0x41]
qc.syncwf 30

# CHECK-INST: qc.syncwl    6
# CHECK-ENC: encoding: [0x13,0x30,0x60,0x80]
qc.syncwl 6

# CHECK-NOALIAS: qc.c.sync      0
# CHECK-ALIAS: qc.sync 0
# CHECK-ENC: encoding: [0x01,0x80]
qc.c.sync 0

# CHECK-NOALIAS: qc.c.syncr     15
# CHECK-ALIAS: qc.syncr 15
# CHECK-ENC: encoding: [0x01,0x87]
qc.c.syncr 15

# CHECK-NOALIAS: qc.c.syncwf    31
# CHECK-ALIAS: qc.syncwf 31
# CHECK-ENC: encoding: [0x81,0x93]
qc.c.syncwf 31

# CHECK-NOALIAS: qc.c.syncwl    4
# CHECK-ALIAS: qc.syncwl 4
# CHECK-ENC: encoding: [0x81,0x95]
qc.c.syncwl 4

# Check that compressed patterns work

# CHECK-NOALIAS: qc.c.sync      8
# CHECK-ALIAS: qc.sync 8
# CHECK-ENC: encoding: [0x01,0x82]
qc.sync 8

# CHECK-NOALIAS: qc.c.syncr     31
# CHECK-ALIAS: qc.syncr 31
# CHECK-ENC: encoding: [0x81,0x87]
qc.syncr 31

# CHECK-NOALIAS: qc.c.syncwf    0
# CHECK-ALIAS: qc.syncwf 0
# CHECK-ENC: encoding: [0x01,0x90]
qc.syncwf 0

# CHECK-NOALIAS: qc.c.syncwl    16
# CHECK-ALIAS: qc.syncwl 16
# CHECK-ENC: encoding: [0x81,0x96]
qc.syncwl 16
