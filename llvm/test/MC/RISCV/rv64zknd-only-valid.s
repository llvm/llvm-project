# RUN: llvm-mc %s -triple=riscv64 -mattr=+zknd -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+zknd < %s \
# RUN:     | llvm-objdump --mattr=+zknd --no-print-imm-hex -d -r - \
# RUN:     | FileCheck --check-prefix=CHECK-ASM-AND-OBJ %s

# CHECK-ASM-AND-OBJ: aes64ds a0, a1, a2
# CHECK-ASM: [0x33,0x85,0xc5,0x3a]
aes64ds a0, a1, a2

# CHECK-ASM-AND-OBJ: aes64dsm a0, a1, a2
# CHECK-ASM: [0x33,0x85,0xc5,0x3e]
aes64dsm a0, a1, a2

# CHECK-ASM-AND-OBJ: aes64im a0, a1
# CHECK-ASM: [0x13,0x95,0x05,0x30]
aes64im a0, a1

# CHECK-ASM-AND-OBJ: aes64ks1i a0, a1, 5
# CHECK-ASM: [0x13,0x95,0x55,0x31]
aes64ks1i a0, a1, 5

# CHECK-ASM-AND-OBJ: aes64ks2 a0, a1, a2
# CHECK-ASM: [0x33,0x85,0xc5,0x7e]
aes64ks2 a0, a1, a2
