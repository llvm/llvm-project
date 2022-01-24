# RUN: llvm-mc %s -triple=riscv64 -mattr=+zkne -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+zkne < %s \
# RUN:     | llvm-objdump --mattr=+zkne -d -r - \
# RUN:     | FileCheck --check-prefix=CHECK-ASM-AND-OBJ %s

# CHECK-ASM-AND-OBJ: aes64es a0, a1, a2
# CHECK-ASM: [0x33,0x85,0xc5,0x32]
aes64es a0, a1, a2

# CHECK-ASM-AND-OBJ: aes64esm a0, a1, a2
# CHECK-ASM: [0x33,0x85,0xc5,0x36]
aes64esm a0, a1, a2

# CHECK-ASM-AND-OBJ: aes64ks1i a0, a1, 5
# CHECK-ASM: [0x13,0x95,0x55,0x31]
aes64ks1i a0, a1, 5

# CHECK-ASM-AND-OBJ: aes64ks2 a0, a1, a2
# CHECK-ASM: [0x33,0x85,0xc5,0x7e]
aes64ks2 a0, a1, a2
