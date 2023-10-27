# With XTHeadBa (address generation) extension:
# RUN: llvm-mc %s -triple=riscv32 -mattr=+xtheadba -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc %s -triple=riscv64 -mattr=+xtheadba -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv32 -mattr=+xtheadba < %s \
# RUN:     | llvm-objdump --mattr=+xtheadba -d -r - \
# RUN:     | FileCheck --check-prefix=CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+xtheadba < %s \
# RUN:     | llvm-objdump --mattr=+xtheadba -d -r - \
# RUN:     | FileCheck --check-prefix=CHECK-ASM-AND-OBJ %s

# CHECK-ASM-AND-OBJ: th.addsl t0, t1, t2, 0
# CHECK-ASM: encoding: [0x8b,0x12,0x73,0x00]
th.addsl t0, t1, t2, 0
# CHECK-ASM-AND-OBJ: th.addsl t0, t1, t2, 1
# CHECK-ASM: encoding: [0x8b,0x12,0x73,0x02]
th.addsl t0, t1, t2, 1
# CHECK-ASM-AND-OBJ: th.addsl t0, t1, t2, 2
# CHECK-ASM: encoding: [0x8b,0x12,0x73,0x04]
th.addsl t0, t1, t2, 2
# CHECK-ASM-AND-OBJ: th.addsl t0, t1, t2, 3
# CHECK-ASM: encoding: [0x8b,0x12,0x73,0x06]
th.addsl t0, t1, t2, 3
