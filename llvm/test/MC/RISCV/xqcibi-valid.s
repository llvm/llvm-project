# Xqcibi - Qualcomm uC Branch Immediate Extension
# RUN: llvm-mc %s -triple=riscv32 -mattr=+experimental-xqcibi -riscv-no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ENC,CHECK-INST %s
# RUN: llvm-mc -filetype=obj -triple riscv32 -mattr=+experimental-xqcibi < %s \
# RUN:     | llvm-objdump --mattr=+experimental-xqcibi -M no-aliases -d - \
# RUN:     | FileCheck -check-prefix=CHECK-OBJ %s
# RUN: llvm-mc %s -triple=riscv32 -mattr=+experimental-xqcibi -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ENC,CHECK-INST %s
# RUN: llvm-mc -filetype=obj -triple riscv32 -mattr=+experimental-xqcibi < %s \
# RUN:     | llvm-objdump --mattr=+experimental-xqcibi -d - \
# RUN:     | FileCheck -check-prefix=CHECK-OBJ %s

# CHECK-INST:  qc.beqi     s0, 12, 346
# CHECK-OBJ:  qc.beqi     s0, 0xc, 0x15a
# CHECK-ENC: encoding: [0x7b,0x0d,0xc4,0x14]
qc.beqi x8, 12, 346

# CHECK-INST: qc.bnei     tp, 15, 4094
# CHECK-OBJ: qc.bnei     tp, 0xf, 0x1002
# CHECK-ENC: encoding: [0xfb,0x1f,0xf2,0x7e]
qc.bnei x4, 15, 4094

# CHECK-INST: qc.bgei     a0, 1, -4096
# CHECK-OBJ: qc.bgei     a0, 0x1, 0xfffff008
# CHECK-ENC: encoding: [0x7b,0x50,0x15,0x80]
qc.bgei x10, 1, -4096

# CHECK-INST: qc.blti     ra, 6, 2000
# CHECK-OBJ: qc.blti     ra, 0x6, 0x7dc
# CHECK-ENC: encoding: [0x7b,0xc8,0x60,0x7c]
qc.blti x1, 6, 2000

# CHECK-INST: qc.bgeui    a2, 11, 128
# CHECK-OBJ: qc.bgeui    a2, 0xb, 0x90
# CHECK-ENC: encoding: [0x7b,0x70,0xb6,0x08]
qc.bgeui x12, 11, 128

# CHECK-INST: qc.bltui    sp, 7, 666
# CHECK-OBJ: qc.bltui    sp, 0x7, 0x2ae
# CHECK-ENC: encoding: [0x7b,0x6d,0x71,0x28]
qc.bltui x2, 7, 666

# CHECK-INST:  qc.e.beqi   ra, 1, 2
# CHECK-OBJ:  qc.e.beqi   ra, 0x1, 0x1a
# CHECK-ENC: encoding: [0x1f,0xc1,0x80,0x01,0x01,0x00]
qc.e.beqi x1, 1, 2

# CHECK-INST: qc.e.bnei   tp, 115, 4094
# CHECK-OBJ: qc.e.bnei   tp, 0x73, 0x101c
# CHECK-ENC: encoding: [0x9f,0x4f,0x92,0x7f,0x73,0x00]
qc.e.bnei x4, 115, 4094

# CHECK-INST: qc.e.bgei   a0, -32768, -4096
# CHECK-OBJ: qc.e.bgei   a0, -0x8000, 0xfffff024
# CHECK-ENC: encoding: [0x1f,0x40,0xd5,0x81,0x00,0x80]
qc.e.bgei x10, -32768, -4096

# CHECK-INST: qc.e.blti   ra, 32767, 2000
# CHECK-OBJ: qc.e.blti   ra, 0x7fff, 0x7fa
# CHECK-ENC: encoding: [0x1f,0xc8,0xc0,0x7d,0xff,0x7f]
qc.e.blti x1, 32767, 2000

# CHECK-INST: qc.e.bgeui  a2, 711, 128
# CHECK-OBJ: qc.e.bgeui  a2, 0x2c7, 0xb0
# CHECK-ENC: encoding: [0x1f,0x40,0xf6,0x09,0xc7,0x02]
qc.e.bgeui x12, 711, 128

# CHECK-INST: qc.e.bltui  sp, 7, 666
# CHECK-OBJ: qc.e.bltui  sp, 0x7, 0x2d0
# CHECK-ENC: encoding: [0x1f,0x4d,0xe1,0x29,0x07,0x00]
qc.e.bltui x2, 7, 666
