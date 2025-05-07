# Xqcili - Qualcomm uC Load Large Immediate Extension
# RUN: llvm-mc %s -triple=riscv32 -mattr=+experimental-xqcili -M no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ENC,CHECK-INST %s

# RUN: llvm-mc -filetype=obj -triple riscv32 -mattr=+experimental-xqcili < %s \
# RUN:     | llvm-objdump --mattr=+experimental-xqcili -M no-aliases --no-print-imm-hex -d - \
# RUN:     | FileCheck -check-prefix=CHECK-INST %s

# RUN: llvm-mc %s -triple=riscv32 -mattr=+experimental-xqcili -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ENC,CHECK-INST %s

# RUN: llvm-mc -filetype=obj -triple riscv32 -mattr=+experimental-xqcili < %s \
# RUN:     | llvm-objdump --mattr=+experimental-xqcili --no-print-imm-hex -d - \
# RUN:     | FileCheck -check-prefix=CHECK-INST %s

# CHECK-INST: qc.e.li a0, -1
# CHECK-ENC: encoding: [0x1f,0x05,0xff,0xff,0xff,0xff]
qc.e.li x10, 4294967295

# CHECK-INST: qc.e.li a0, -2147483648
# CHECK-ENC: encoding: [0x1f,0x05,0x00,0x00,0x00,0x80]
qc.e.li x10, -2147483648

# CHECK-INST: qc.e.li s1, -33554432
# CHECK-ENC: encoding: [0x9f,0x04,0x00,0x00,0x00,0xfe]
qc.e.li x9, -33554432

# CHECK-INST: qc.e.li s1, 33554431
# CHECK-ENC: encoding: [0x9f,0x04,0xff,0xff,0xff,0x01]
qc.e.li x9, 33554431

# CHECK-INST: qc.li   s1, 524287
# CHECK-ENC: encoding: [0x9b,0xf4,0xff,0x7f]
qc.li x9, 524287

# CHECK-INST: qc.li   s1, -524288
# CHECK-ENC: encoding: [0x9b,0x04,0x00,0x80]
qc.li x9, -524288

# CHECK-INST: qc.li   a0, 12345
# CHECK-ENC: encoding: [0x1b,0x05,0x39,0x30]
qc.li x10, 12345

# CHECK-INST: qc.li   a0, -12346
# CHECK-ENC: encoding: [0x1b,0xf5,0xc6,0xcf]
qc.li x10, -12346
