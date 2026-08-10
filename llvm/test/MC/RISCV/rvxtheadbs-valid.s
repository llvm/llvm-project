# RUN: llvm-mc %s -triple=riscv32 -mattr=+xtheadbs -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc %s -triple=riscv64 -mattr=+xtheadbs -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv32 -mattr=+xtheadbs < %s \
# RUN:     | llvm-objdump --mattr=+xtheadbs --no-print-imm-hex -d -r - \
# RUN:     | FileCheck --check-prefix=CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+xtheadbs < %s \
# RUN:     | llvm-objdump --mattr=+xtheadbs --no-print-imm-hex -d -r - \
# RUN:     | FileCheck --check-prefix=CHECK-ASM-AND-OBJ %s

# CHECK-ASM-AND-OBJ: th.tst t0, t1, 1
# CHECK-ASM: encoding: [0x8b,0x12,0x13,0x88]
th.tst t0, t1, 1
# CHECK-ASM-AND-OBJ: th.tst t0, t1, 31
# CHECK-ASM: encoding: [0x8b,0x12,0xf3,0x89]
th.tst t0, t1, 31
