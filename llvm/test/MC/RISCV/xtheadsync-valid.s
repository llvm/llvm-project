# RUN: llvm-mc %s -triple=riscv32 -mattr=+xtheadsync -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv32 -mattr=+xtheadsync < %s \
# RUN:     | llvm-objdump --mattr=+xtheadsync -d -r - \
# RUN:     | FileCheck --check-prefix=CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc %s -triple=riscv64 -mattr=+xtheadsync -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+xtheadsync < %s \
# RUN:     | llvm-objdump --mattr=+xtheadsync -d -r - \
# RUN:     | FileCheck --check-prefix=CHECK-ASM-AND-OBJ %s

# CHECK-ASM-AND-OBJ: th.sfence.vmas a0, a1
# CHECK-ASM: encoding: [0x0b,0x00,0xb5,0x04]
th.sfence.vmas a0, a1

# CHECK-ASM-AND-OBJ: th.sync
# CHECK-ASM: encoding: [0x0b,0x00,0x80,0x01]
th.sync

# CHECK-ASM-AND-OBJ: th.sync.i
# CHECK-ASM: encoding: [0x0b,0x00,0xa0,0x01]
th.sync.i

# CHECK-ASM-AND-OBJ: th.sync.is
# CHECK-ASM: encoding: [0x0b,0x00,0xb0,0x01]
th.sync.is

# CHECK-ASM-AND-OBJ: th.sync.s
# CHECK-ASM: encoding: [0x0b,0x00,0x90,0x01]
th.sync.s
