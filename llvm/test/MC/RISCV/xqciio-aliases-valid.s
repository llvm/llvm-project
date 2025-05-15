# Xqciio - Qualcomm uC External Input Output extension
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

# CHECK-INST: qc.outw t0, 0(a0)
# CHECK-ENC: encoding: [0x8b,0x42,0x05,0x00]
qc.outw x5, (x10)


# CHECK-INST: qc.inw  t0, 0(a0)
# CHECK-ENC: encoding: [0x8b,0x52,0x05,0x00]
qc.inw x5, (x10)
