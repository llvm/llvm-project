# RUN: llvm-mc -triple=riscv32 -mattr=+c -riscv-no-aliases -show-encoding < %s \
# RUN:     | FileCheck -check-prefixes=CHECK,CHECK-INST %s
# RUN: llvm-mc -filetype=obj -triple riscv32 -mattr=+c -riscv-no-aliases < %s \
# RUN:     | llvm-objdump -mattr=+c -d - | FileCheck -check-prefix=CHECK-INST %s

# CHECK-INST: c.jal    2046
# CHECK: encoding: [0xfd,0x2f]
c.jal    2046
