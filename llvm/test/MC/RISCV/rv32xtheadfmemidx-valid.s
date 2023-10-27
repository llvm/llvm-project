# RUN: llvm-mc %s -triple=riscv32 -mattr=+d -mattr=+xtheadfmemidx -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv32 -mattr=+d -mattr=+xtheadfmemidx < %s \
# RUN:     | llvm-objdump --mattr=+d --mattr=+xtheadfmemidx -d -r - \
# RUN:     | FileCheck --check-prefix=CHECK-ASM-AND-OBJ %s

# CHECK-ASM-AND-OBJ: th.flrd fa0, a1, a2, 0
# CHECK-ASM: encoding: [0x0b,0xe5,0xc5,0x60]
th.flrd fa0, a1, a2, 0

# CHECK-ASM-AND-OBJ: th.flrd fa0, a1, a2, 3
# CHECK-ASM: encoding: [0x0b,0xe5,0xc5,0x66]
th.flrd fa0, a1, a2, 3

# CHECK-ASM-AND-OBJ: th.flrw fa0, a1, a2, 0
# CHECK-ASM: encoding: [0x0b,0xe5,0xc5,0x40]
th.flrw fa0, a1, a2, 0

# CHECK-ASM-AND-OBJ: th.flrw fa0, a1, a2, 3
# CHECK-ASM: encoding: [0x0b,0xe5,0xc5,0x46]
th.flrw fa0, a1, a2, 3

# CHECK-ASM-AND-OBJ: th.fsrd fa0, a1, a2, 0
# CHECK-ASM: encoding: [0x0b,0xf5,0xc5,0x60]
th.fsrd fa0, a1, a2, 0

# CHECK-ASM-AND-OBJ: th.fsrd fa0, a1, a2, 3
# CHECK-ASM: encoding: [0x0b,0xf5,0xc5,0x66]
th.fsrd fa0, a1, a2, 3

# CHECK-ASM-AND-OBJ: th.fsrw fa0, a1, a2, 0
# CHECK-ASM: encoding: [0x0b,0xf5,0xc5,0x40]
th.fsrw fa0, a1, a2, 0

# CHECK-ASM-AND-OBJ: th.fsrw fa0, a1, a2, 3
# CHECK-ASM: encoding: [0x0b,0xf5,0xc5,0x46]
th.fsrw fa0, a1, a2, 3
