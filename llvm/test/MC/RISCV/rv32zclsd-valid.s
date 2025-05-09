# RUN: llvm-mc %s -triple=riscv32 -mattr=+zclsd -M no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv32 -mattr=+zclsd< %s \
# RUN:     | llvm-objdump --mattr=+zclsd --no-print-imm-hex -M no-aliases -d -r - \
# RUN:     | FileCheck --check-prefix=CHECK-ASM-AND-OBJ %s

# CHECK-ASM-AND-OBJ: c.ldsp t1, 176(sp)
# CHECK-ASM: encoding: [0x4a,0x73]
c.ldsp t1, 176(sp)
# CHECK-ASM-AND-OBJ: c.sdsp t1, 360(sp)
# CHECK-ASM: encoding: [0x9a,0xf6]
c.sdsp t1, 360(sp)
# CHECK-ASM-AND-OBJ: c.ld a4, 0(a3)
# CHECK-ASM: encoding: [0x98,0x62]
c.ld a4, 0(a3)
# CHECK-ASM-AND-OBJ: c.sd s0, 248(a3)
# CHECK-ASM: encoding: [0xe0,0xfe]
c.sd s0, 248(a3)
