# RUN: llvm-mc %s -triple=riscv32 -mattr=+zcmop -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc %s -triple=riscv64 -mattr=+zcmop -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv32 -mattr=+zcmop < %s \
# RUN:     | llvm-objdump --mattr=+zcmop -d -r - \
# RUN:     | FileCheck --check-prefix=CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+zcmop < %s \
# RUN:     | llvm-objdump --mattr=+zcmop -d -r - \
# RUN:     | FileCheck --check-prefix=CHECK-ASM-AND-OBJ %s

# CHECK-ASM-AND-OBJ: c.mop.1
# CHECK-ASM: encoding: [0x81,0x60]
c.mop.1

# CHECK-ASM-AND-OBJ: c.mop.3
# CHECK-ASM: encoding: [0x81,0x61]
c.mop.3

# CHECK-ASM-AND-OBJ: c.mop.5
# CHECK-ASM: encoding: [0x81,0x62]
c.mop.5

# CHECK-ASM-AND-OBJ: c.mop.7
# CHECK-ASM: encoding: [0x81,0x63]
c.mop.7

# CHECK-ASM-AND-OBJ: c.mop.9
# CHECK-ASM: encoding: [0x81,0x64]
c.mop.9

# CHECK-ASM-AND-OBJ: c.mop.11
# CHECK-ASM: encoding: [0x81,0x65]
c.mop.11

# CHECK-ASM-AND-OBJ: c.mop.13
# CHECK-ASM: encoding: [0x81,0x66]
c.mop.13

# CHECK-ASM-AND-OBJ: c.mop.15
# CHECK-ASM: encoding: [0x81,0x67]
c.mop.15
