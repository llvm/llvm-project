# Xqcilsm - Qualcomm uC Load Store Multiple Extension
# RUN: llvm-mc %s -triple=riscv32 -mattr=+experimental-xqcilsm -M no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ENC,CHECK-INST %s
# RUN: llvm-mc -filetype=obj -triple riscv32 -mattr=+experimental-xqcilsm < %s \
# RUN:     | llvm-objdump --mattr=+experimental-xqcilsm -M no-aliases --no-print-imm-hex -d - \
# RUN:     | FileCheck -check-prefix=CHECK-INST %s
# RUN: llvm-mc %s -triple=riscv32 -mattr=+experimental-xqcilsm -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ENC,CHECK-INST %s
# RUN: llvm-mc -filetype=obj -triple riscv32 -mattr=+experimental-xqcilsm < %s \
# RUN:     | llvm-objdump --mattr=+experimental-xqcilsm --no-print-imm-hex -d - \
# RUN:     | FileCheck -check-prefix=CHECK-INST %s

# CHECK-INST: qc.swm   t0, s4, 0(gp)
# CHECK-ENC: encoding: [0xab,0xf2,0x41,0x01]
qc.swm x5, x20, (x3)

# CHECK-INST: qc.swmi  a0, 4, 0(tp)
# CHECK-ENC: encoding: [0x2b,0x75,0x42,0x40]
qc.swmi x10, 4, (x4)

# CHECK-INST: qc.setwm tp, t5, 0(sp)
# CHECK-ENC: encoding: [0x2b,0x72,0xe1,0x81]
qc.setwm x4, x30, (x2)

# CHECK-INST: qc.setwmi    t0, 31, 0(a2)
# CHECK-ENC: encoding: [0xab,0x72,0xf6,0xc1]
qc.setwmi x5, 31, (x12)

# CHECK-INST: qc.lwm   t2, ra, 0(s4)
# CHECK-ENC: encoding: [0x8b,0x73,0x1a,0x00]
qc.lwm x7, x1, (x20)

# CHECK-INST: qc.lwmi  a3, 9, 0(s7)
# CHECK-ENC: encoding: [0x8b,0xf6,0x9b,0x40]
qc.lwmi x13, 9, (x23)
