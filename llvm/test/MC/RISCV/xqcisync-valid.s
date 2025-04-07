# Xqcisync - Qualcomm uC Sync Delay Extension
# RUN: llvm-mc %s -triple=riscv32 -mattr=+experimental-xqcisync -riscv-no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ENC,CHECK-INST %s
# RUN: llvm-mc -filetype=obj -triple riscv32 -mattr=+experimental-xqcisync < %s \
# RUN:     | llvm-objdump --mattr=+experimental-xqcisync -M no-aliases --no-print-imm-hex -d - \
# RUN:     | FileCheck -check-prefix=CHECK-INST %s
# RUN: llvm-mc %s -triple=riscv32 -mattr=+experimental-xqcisync -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ENC,CHECK-INST %s
# RUN: llvm-mc -filetype=obj -triple riscv32 -mattr=+experimental-xqcisync < %s \
# RUN:     | llvm-objdump --mattr=+experimental-xqcisync --no-print-imm-hex -d - \
# RUN:     | FileCheck -check-prefix=CHECK-INST %s

# CHECK-INST: qc.c.delay     10
# CHECK-ENC: encoding: [0x2a,0x00]
qc.c.delay 10

# CHECK-INST: qc.sync      8
# CHECK-ENC: encoding: [0x13,0x30,0x80,0x10]
qc.sync 8

# CHECK-INST: qc.syncr     23
# CHECK-ENC: encoding: [0x13,0x30,0x70,0x21]
qc.syncr 23

# CHECK-INST: qc.syncwf    31
# CHECK-ENC: encoding: [0x13,0x30,0xf0,0x41]
qc.syncwf 31

# CHECK-INST: qc.syncwl    1
# CHECK-ENC: encoding: [0x13,0x30,0x10,0x80]
qc.syncwl 1

# CHECK-INST: qc.c.sync      0
# CHECK-ENC: encoding: [0x01,0x80]
qc.c.sync 0

# CHECK-INST: qc.c.syncr     15
# CHECK-ENC: encoding: [0x01,0x87]
qc.c.syncr 15

# CHECK-INST: qc.c.syncwf    31
# CHECK-ENC: encoding: [0x81,0x93]
qc.c.syncwf 31

# CHECK-INST: qc.c.syncwl    4
# CHECK-ENC: encoding: [0x81,0x95]
qc.c.syncwl 4
