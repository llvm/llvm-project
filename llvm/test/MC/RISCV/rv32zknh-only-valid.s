# RUN: llvm-mc %s -triple=riscv32 -mattr=+zknh -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv32 -mattr=+zknh < %s \
# RUN:     | llvm-objdump --mattr=+zknh -d -r - \
# RUN:     | FileCheck --check-prefix=CHECK-ASM-AND-OBJ %s

# CHECK-ASM-AND-OBJ: sha512sig0h a0, a1, a2
# CHECK-ASM: [0x33,0x85,0xc5,0x5c]
sha512sig0h a0, a1, a2

# CHECK-ASM-AND-OBJ: sha512sig1h a0, a1, a2
# CHECK-ASM: [0x33,0x85,0xc5,0x5e]
sha512sig1h a0, a1, a2

# CHECK-ASM-AND-OBJ: sha512sig0l a0, a1, a2
# CHECK-ASM: [0x33,0x85,0xc5,0x54]
sha512sig0l a0, a1, a2

# CHECK-ASM-AND-OBJ: sha512sig1l a0, a1, a2
# CHECK-ASM: [0x33,0x85,0xc5,0x56]
sha512sig1l a0, a1, a2

# CHECK-ASM-AND-OBJ: sha512sum0r a0, a1, a2
# CHECK-ASM: [0x33,0x85,0xc5,0x50]
sha512sum0r a0, a1, a2

# CHECK-ASM-AND-OBJ: sha512sum1r a0, a1, a2
# CHECK-ASM: [0x33,0x85,0xc5,0x52]
sha512sum1r a0, a1, a2
