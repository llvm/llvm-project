# With Bitmanip base extension:
# RUN: llvm-mc %s -triple=riscv32 -mattr=+xtheadmempair -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv32 -mattr=+xtheadmempair < %s \
# RUN:     | llvm-objdump --mattr=+xtheadmempair -d -r - \
# RUN:     | FileCheck --check-prefix=CHECK-ASM-AND-OBJ %s

# CHECK-ASM-AND-OBJ: th.lwd
# CHECK-ASM: encoding: [0x0b,0x45,0xb6,0xe2]
th.lwd a0, a1, (a2), 1, 3

# CHECK-ASM-AND-OBJ: th.lwud
# CHECK-ASM: encoding: [0x0b,0x45,0xb6,0xf4]
th.lwud a0, a1, (a2), 2, 3

# CHECK-ASM-AND-OBJ: th.swd
# CHECK-ASM: encoding: [0x0b,0x55,0xb6,0xe0]
th.swd a0, a1, (a2), 0, 3
