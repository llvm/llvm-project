# Xqciio - Qualcomm uC External Input Output Extension
# RUN: llvm-mc %s -triple=riscv32 -mattr=+experimental-xqciio -M no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ENC,CHECK-INST %s
# RUN: llvm-mc -filetype=obj -triple riscv32 -mattr=+experimental-xqciio < %s \
# RUN:     | llvm-objdump --mattr=+experimental-xqciio -M no-aliases --no-print-imm-hex -d - \
# RUN:     | FileCheck -check-prefix=CHECK-INST %s
# RUN: llvm-mc %s -triple=riscv32 -mattr=+experimental-xqciio -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ENC,CHECK-INST %s
# RUN: llvm-mc -filetype=obj -triple riscv32 -mattr=+experimental-xqciio < %s \
# RUN:     | llvm-objdump --mattr=+experimental-xqciio --no-print-imm-hex -d - \
# RUN:     | FileCheck -check-prefix=CHECK-INST %s


# CHECK-INST: qc.outw    t0, 2048(a0)
# CHECK-ENC: encoding: [0x8b,0x42,0x05,0x20]
qc.outw x5, 2048(x10)

# CHECK-INST: qc.inw    s7, 16380(a7)
# CHECK-ENC: encoding: [0x8b,0xdb,0xf8,0xff]
qc.inw x23, 16380(x17)
