# RUN: llvm-mc %s -triple=riscv32 -mattr=+zknh -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc %s -triple=riscv64 -mattr=+zknh -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv32 -mattr=+zknh < %s \
# RUN:     | llvm-objdump --mattr=+zknh -d -r - \
# RUN:     | FileCheck --check-prefix=CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+zknh < %s \
# RUN:     | llvm-objdump --mattr=+zknh -d -r - \
# RUN:     | FileCheck --check-prefix=CHECK-ASM-AND-OBJ %s

# CHECK-ASM-AND-OBJ: sha256sig0 a0, a1
# CHECK-ASM: [0x13,0x95,0x25,0x10]
sha256sig0 a0, a1

# CHECK-ASM-AND-OBJ: sha256sig1 a0, a1
# CHECK-ASM: [0x13,0x95,0x35,0x10]
sha256sig1 a0, a1

# CHECK-ASM-AND-OBJ: sha256sum0 a0, a1
# CHECK-ASM: [0x13,0x95,0x05,0x10]
sha256sum0 a0, a1

# CHECK-ASM-AND-OBJ: sha256sum1 a0, a1
# CHECK-ASM: [0x13,0x95,0x15,0x10]
sha256sum1 a0, a1
