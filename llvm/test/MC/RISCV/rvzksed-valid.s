# RUN: llvm-mc %s -triple=riscv32 -mattr=+zksed -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc %s -triple=riscv64 -mattr=+zksed -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv32 -mattr=+zksed < %s \
# RUN:     | llvm-objdump --mattr=+zksed --no-print-imm-hex -d -r - \
# RUN:     | FileCheck --check-prefix=CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+zksed < %s \
# RUN:     | llvm-objdump --mattr=+zksed --no-print-imm-hex -d -r - \
# RUN:     | FileCheck --check-prefix=CHECK-ASM-AND-OBJ %s

# CHECK-ASM-AND-OBJ: sm4ed a0, a1, a2, 3
# CHECK-ASM: [0x33,0x85,0xc5,0xf0]
sm4ed a0, a1, a2, 3

# CHECK-ASM-AND-OBJ: sm4ks a0, a1, a2, 3
# CHECK-ASM: [0x33,0x85,0xc5,0xf4]
sm4ks a0, a1, a2, 3
