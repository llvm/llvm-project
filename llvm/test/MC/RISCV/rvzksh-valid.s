# RUN: llvm-mc %s -triple=riscv32 -mattr=+zksh -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc %s -triple=riscv64 -mattr=+zksh -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv32 -mattr=+zksh < %s \
# RUN:     | llvm-objdump --mattr=+zksh -d -r - \
# RUN:     | FileCheck --check-prefix=CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+zksh < %s \
# RUN:     | llvm-objdump --mattr=+zksh -d -r - \
# RUN:     | FileCheck --check-prefix=CHECK-ASM-AND-OBJ %s

# CHECK-ASM-AND-OBJ: sm3p0 a0, a1
# CHECK-ASM: [0x13,0x95,0x85,0x10]
sm3p0 a0, a1

# CHECK-ASM-AND-OBJ: sm3p1 a0, a1
# CHECK-ASM: [0x13,0x95,0x95,0x10]
sm3p1 a0, a1
