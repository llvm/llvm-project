# RUN: llvm-mc %s -triple=riscv32 -mattr=+xtheadcondmov -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv32 -mattr=+xtheadcondmov < %s \
# RUN:     | llvm-objdump --mattr=+xtheadcondmov -d -r - \
# RUN:     | FileCheck --check-prefix=CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc %s -triple=riscv64 -mattr=+xtheadcondmov -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+xtheadcondmov < %s \
# RUN:     | llvm-objdump --mattr=+xtheadcondmov -d -r - \
# RUN:     | FileCheck --check-prefix=CHECK-ASM-AND-OBJ %s

# CHECK-ASM-AND-OBJ: th.mveqz a0, a1, a2
# CHECK-ASM: encoding: [0x0b,0x95,0xc5,0x40]
th.mveqz a0,a1,a2

# CHECK-ASM-AND-OBJ: th.mvnez a0, a1, a2
# CHECK-ASM: encoding: [0x0b,0x95,0xc5,0x42]
th.mvnez a0,a1,a2
