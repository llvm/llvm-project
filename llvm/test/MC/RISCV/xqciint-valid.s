# Xqciint - Qualcomm uC Interrupts extension
# RUN: llvm-mc %s -triple=riscv32 -mattr=+experimental-xqciint -riscv-no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ENC,CHECK-INST %s
# RUN: llvm-mc -filetype=obj -triple riscv32 -mattr=+experimental-xqciint < %s \
# RUN:     | llvm-objdump --mattr=+experimental-xqciint -M no-aliases --no-print-imm-hex -d - \
# RUN:     | FileCheck -check-prefix=CHECK-INST %s
# RUN: llvm-mc %s -triple=riscv32 -mattr=+experimental-xqciint -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ENC,CHECK-INST %s
# RUN: llvm-mc -filetype=obj -triple riscv32 -mattr=+experimental-xqciint < %s \
# RUN:     | llvm-objdump --mattr=+experimental-xqciint --no-print-imm-hex -d - \
# RUN:     | FileCheck -check-prefix=CHECK-INST %s

# CHECK-INST: qc.setinti      500
# CHECK-ENC: encoding: [0x73,0x00,0xfa,0xcc]
qc.setinti 500

# CHECK-INST: qc.setinti      0
# CHECK-ENC: encoding: [0x73,0x00,0x00,0xcc]
qc.setinti 0

# CHECK-INST: qc.setinti      1023
# CHECK-ENC: encoding: [0x73,0x80,0xff,0xcd]
qc.setinti 1023


# CHECK-INST: qc.clrinti      500
# CHECK-ENC: encoding: [0x73,0x00,0xfa,0xce]
qc.clrinti 500

# CHECK-INST: qc.clrinti      1023
# CHECK-ENC: encoding: [0x73,0x80,0xff,0xcf]
qc.clrinti 1023

# CHECK-INST: qc.clrinti      0
# CHECK-ENC: encoding: [0x73,0x00,0x00,0xce]
qc.clrinti 0


# CHECK-INST: qc.c.clrint     a0
# CHECK-ENC: encoding: [0x0e,0x15]
qc.c.clrint x10


# CHECK-INST: qc.c.di
# CHECK-ENC: encoding: [0x12,0x1b]
qc.c.di


# CHECK-INST: qc.c.dir        a0
# CHECK-ENC: encoding: [0x02,0x15]
qc.c.dir x10


# CHECK-INST: qc.c.ei
# CHECK-ENC: encoding: [0x92,0x1b]
qc.c.ei


# CHECK-INST: qc.c.eir        a0
# CHECK-ENC: encoding: [0x06,0x15]
qc.c.eir x10


# CHECK-INST: qc.c.mienter.nest
# CHECK-ENC: encoding: [0x92,0x18]
qc.c.mienter.nest


# CHECK-INST: qc.c.mienter
# CHECK-ENC: encoding: [0x12,0x18]
qc.c.mienter


# CHECK-INST: qc.c.mileaveret
# CHECK-ENC: encoding: [0x12,0x1a]
qc.c.mileaveret


# CHECK-INST: qc.c.setint     a0
# CHECK-ENC: encoding: [0x0a,0x15]
qc.c.setint x10
