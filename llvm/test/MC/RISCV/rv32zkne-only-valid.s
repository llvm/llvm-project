# RUN: llvm-mc %s -triple=riscv32 -mattr=+zkne -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv32 -mattr=+zkne < %s \
# RUN:     | llvm-objdump --mattr=+zkne --no-print-imm-hex -d -r - \
# RUN:     | FileCheck --check-prefix=CHECK-ASM-AND-OBJ %s

# CHECK-ASM-AND-OBJ: aes32esi a0, a1, a2, 3
# CHECK-ASM: [0x33,0x85,0xc5,0xe2]
aes32esi a0, a1, a2, 3

# CHECK-ASM-AND-OBJ: aes32esmi a0, a1, a2, 3
# CHECK-ASM: [0x33,0x85,0xc5,0xe6]
aes32esmi a0, a1, a2, 3
