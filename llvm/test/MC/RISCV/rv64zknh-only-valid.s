# RUN: llvm-mc %s -triple=riscv64 -mattr=+zknh -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+zknh < %s \
# RUN:     | llvm-objdump --mattr=+zknh -d -r - \
# RUN:     | FileCheck --check-prefix=CHECK-ASM-AND-OBJ %s

# CHECK-ASM-AND-OBJ: sha512sig0 a0, a1
# CHECK-ASM: [0x13,0x95,0x65,0x10]
sha512sig0 a0, a1

# CHECK-ASM-AND-OBJ: sha512sig1 a0, a1
# CHECK-ASM: [0x13,0x95,0x75,0x10]
sha512sig1 a0, a1

# CHECK-ASM-AND-OBJ: sha512sum0 a0, a1
# CHECK-ASM: [0x13,0x95,0x45,0x10]
sha512sum0 a0, a1

# CHECK-ASM-AND-OBJ: sha512sum1 a0, a1
# CHECK-ASM: [0x13,0x95,0x55,0x10]
sha512sum1 a0, a1
