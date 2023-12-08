# RUN: llvm-mc %s -triple=riscv32 -mattr=+zknd -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv32 -mattr=+zknd < %s \
# RUN:     | llvm-objdump --mattr=+zknd -d -r - \
# RUN:     | FileCheck --check-prefix=CHECK-ASM-AND-OBJ %s

# CHECK-ASM-AND-OBJ: aes32dsi a0, a1, a2, 3
# CHECK-ASM: [0x33,0x85,0xc5,0xea]
aes32dsi a0, a1, a2, 3

# CHECK-ASM-AND-OBJ: aes32dsmi a0, a1, a2, 3
# CHECK-ASM: [0x33,0x85,0xc5,0xee]
aes32dsmi a0, a1, a2, 3
