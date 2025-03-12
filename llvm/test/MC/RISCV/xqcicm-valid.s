# Xqcicm - Qualcomm uC Conditional Move Extension
# RUN: llvm-mc %s -triple=riscv32 -mattr=+experimental-xqcicm -riscv-no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ENC,CHECK-INST %s
# RUN: llvm-mc -filetype=obj -triple riscv32 -mattr=+experimental-xqcicm < %s \
# RUN:     | llvm-objdump --mattr=+experimental-xqcicm -M no-aliases --no-print-imm-hex -d - \
# RUN:     | FileCheck -check-prefix=CHECK-INST %s
# RUN: llvm-mc %s -triple=riscv32 -mattr=+experimental-xqcicm -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ENC,CHECK-INST %s
# RUN: llvm-mc -filetype=obj -triple riscv32 -mattr=+experimental-xqcicm < %s \
# RUN:     | llvm-objdump --mattr=+experimental-xqcicm --no-print-imm-hex -d - \
# RUN:     | FileCheck -check-prefix=CHECK-INST %s

# CHECK-INST: qc.c.mveqz      s1, a0
# CHECK-ENC: encoding: [0x06,0xad]
qc.c.mveqz x9, x10


# CHECK-INST: qc.mveq s1, a0, a1, a2
# CHECK-ENC: encoding: [0xdb,0x04,0xb5,0x60]
qc.mveq x9, x10, x11, x12


# CHECK-INST: qc.mvge s1, a0, a1, a2
# CHECK-ENC: encoding: [0xdb,0x54,0xb5,0x60]
qc.mvge x9, x10, x11, x12


# CHECK-INST: qc.mvgeu        s1, a0, a1, a2
# CHECK-ENC: encoding: [0xdb,0x74,0xb5,0x60]
qc.mvgeu x9, x10, x11, x12


# CHECK-INST: qc.mvlt s1, a0, a1, a2
# CHECK-ENC: encoding: [0xdb,0x44,0xb5,0x60]
qc.mvlt x9, x10, x11, x12


# CHECK-INST: qc.mvltu        s1, a0, a1, a2
# CHECK-ENC: encoding: [0xdb,0x64,0xb5,0x60]
qc.mvltu x9, x10, x11, x12


# CHECK-INST: qc.mvne s1, a0, a1, a2
# CHECK-ENC: encoding: [0xdb,0x14,0xb5,0x60]
qc.mvne x9, x10, x11, x12


# CHECK-INST: qc.mveqi        s1, a0, 5, a2
# CHECK-ENC: encoding: [0xdb,0x04,0x55,0x64]
qc.mveqi x9, x10, 5, x12

# CHECK-INST: qc.mveqi        s1, a0, -16, a2
# CHECK-ENC: encoding: [0xdb,0x04,0x05,0x65]
qc.mveqi x9, x10, -16, x12

# CHECK-INST: qc.mveqi        s1, a0, 15, a2
# CHECK-ENC: encoding: [0xdb,0x04,0xf5,0x64]
qc.mveqi x9, x10, 15, x12


# CHECK-INST: qc.mvgei        s1, a0, 5, a2
# CHECK-ENC: encoding: [0xdb,0x54,0x55,0x64]
qc.mvgei x9, x10, 5, x12

# CHECK-INST: qc.mvgei        s1, a0, -16, a2
# CHECK-ENC: encoding: [0xdb,0x54,0x05,0x65]
qc.mvgei x9, x10, -16, x12

# CHECK-INST: qc.mvgei        s1, a0, 15, a2
# CHECK-ENC: encoding: [0xdb,0x54,0xf5,0x64]
qc.mvgei x9, x10, 15, x12


# CHECK-INST: qc.mvlti        s1, a0, 5, a2
# CHECK-ENC: encoding: [0xdb,0x44,0x55,0x64]
qc.mvlti x9, x10, 5, x12

# CHECK-INST: qc.mvlti        s1, a0, -16, a2
# CHECK-ENC: encoding: [0xdb,0x44,0x05,0x65]
qc.mvlti x9, x10, -16, x12

# CHECK-INST: qc.mvlti        s1, a0, 15, a2
# CHECK-ENC: encoding: [0xdb,0x44,0xf5,0x64]
qc.mvlti x9, x10, 15, x12


# CHECK-INST: qc.mvnei        s1, a0, 5, a2
# CHECK-ENC: encoding: [0xdb,0x14,0x55,0x64]
qc.mvnei x9, x10, 5, x12

# CHECK-INST: qc.mvnei        s1, a0, -16, a2
# CHECK-ENC: encoding: [0xdb,0x14,0x05,0x65]
qc.mvnei x9, x10, -16, x12

# CHECK-INST: qc.mvnei        s1, a0, 15, a2
# CHECK-ENC: encoding: [0xdb,0x14,0xf5,0x64]
qc.mvnei x9, x10, 15, x12


# CHECK-INST: qc.mvltui       s1, a0, 5, a2
# CHECK-ENC: encoding: [0xdb,0x64,0x55,0x64]
qc.mvltui x9, x10, 5, x12

# CHECK-INST: qc.mvltui       s1, a0, 0, a2
# CHECK-ENC: encoding: [0xdb,0x64,0x05,0x64]
qc.mvltui x9, x10, 0, x12

# CHECK-INST: qc.mvltui       s1, a0, 31, a2
# CHECK-ENC: encoding: [0xdb,0x64,0xf5,0x65]
qc.mvltui x9, x10, 31, x12


# CHECK-INST: qc.mvgeui       s1, a0, 5, a2
# CHECK-ENC: encoding: [0xdb,0x74,0x55,0x64]
qc.mvgeui x9, x10, 5, x12

# CHECK-INST: qc.mvgeui       s1, a0, 0, a2
# CHECK-ENC: encoding: [0xdb,0x74,0x05,0x64]
qc.mvgeui x9, x10, 0, x12

# CHECK-INST: qc.mvgeui       s1, a0, 31, a2
# CHECK-ENC: encoding: [0xdb,0x74,0xf5,0x65]
qc.mvgeui x9, x10, 31, x12
