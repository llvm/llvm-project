# Xqcisync - Qualcomm uC Synchronization And Delay Extension
# RUN: llvm-mc -triple=riscv32 -mattr=+experimental-xqcisync < %s -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ENC,CHECK-INST %s
# RUN: llvm-mc -filetype=obj -triple riscv32 -mattr=+experimental-xqcisync < %s \
# RUN:     | llvm-objdump --mattr=+experimental-xqcisync --no-print-imm-hex -d - \
# RUN:     | FileCheck -check-prefix=CHECK-INST %s

# CHECK-INST: qc.c.delay 5
# CHECK-ENC: encoding: [0x16,0x00]
qc.c.delay 5

# CHECK-INST: qc.c.sync 3
# CHECK-ENC: encoding: [0x81,0x81]
qc.c.sync 3

# CHECK-INST: qc.c.syncr 3
# CHECK-ENC: encoding: [0x81,0x85]
qc.c.syncr 3

# CHECK-INST: qc.c.syncwf 5
# CHECK-ENC: encoding: [0x81,0x92]
qc.c.syncwf 5

# CHECK-INST: qc.c.syncwl 7
# CHECK-ENC: encoding: [0x81,0x97]
qc.c.syncwl 7

# CHECK-INST: qc.sync 10
# CHECK-ENC: encoding: [0x13,0x30,0xa0,0x10]
qc.sync 10

# CHECK-INST: qc.syncr 10
# CHECK-ENC: encoding: [0x13,0x30,0xa0,0x20]
qc.syncr 10

# CHECK-INST: qc.syncwf 10
# CHECK-ENC: encoding: [0x13,0x30,0xa0,0x40]
qc.syncwf 10

# CHECK-INST: qc.syncwl 10
# CHECK-ENC: encoding: [0x13,0x30,0xa0,0x80]
qc.syncwl 10
