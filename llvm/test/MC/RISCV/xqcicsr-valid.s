# Xqcicsr - Qualcomm uC CSR Extension
# RUN: llvm-mc %s -triple=riscv32 -mattr=+experimental-xqcicsr -M no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ENC,CHECK-INST %s
# RUN: llvm-mc -filetype=obj -triple riscv32 -mattr=+experimental-xqcicsr < %s \
# RUN:     | llvm-objdump --mattr=+experimental-xqcicsr -M no-aliases --no-print-imm-hex -d - \
# RUN:     | FileCheck -check-prefix=CHECK-INST %s
# RUN: llvm-mc %s -triple=riscv32 -mattr=+experimental-xqcicsr -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ENC,CHECK-INST %s
# RUN: llvm-mc -filetype=obj -triple riscv32 -mattr=+experimental-xqcicsr < %s \
# RUN:     | llvm-objdump --mattr=+experimental-xqcicsr --no-print-imm-hex -d - \
# RUN:     | FileCheck -check-prefix=CHECK-INST %s

# CHECK-INST: qc.csrrwr  a0, t0, s4
# CHECK-ENC: encoding: [0x73,0x85,0x42,0x8d]
qc.csrrwr  x10, x5, x20

# CHECK-INST: qc.csrrwri s4, 31, a2
# CHECK-ENC: encoding: [0x73,0x8a,0xcf,0x8e]
qc.csrrwri x20, 31, x12
