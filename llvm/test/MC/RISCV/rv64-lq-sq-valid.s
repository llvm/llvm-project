# RUN: llvm-mc %s -triple=riscv64 -mattr=+mem128ext -riscv-no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-INST,CHECK-ENC %s
# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+mem128ext < %s \
# RUN:     | llvm-objdump --mattr=+mem128ext -d -r - \
# RUN:     | FileCheck --check-prefix=CHECK-ASM-AND-OBJ %s

# CHECK-ASM-AND-OBJ: sq a1, a4, a0
# CHECK-INST: sq a1, a4, a0
# CHECK-ENC: encoding: [0x0b,0x50,0xe5,0x5a]
sq a1, a4, a0

# CHECK-ASM-AND-OBJ: lq a2, a0, a0
# CHECK-INST: lq a2, a0, a0
# CHECK-ENC: encoding: [0x0b,0x55,0xc5,0x00]
lq a2, a0, a0
