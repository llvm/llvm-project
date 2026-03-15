# RUN: llvm-mc %s -triple=riscv32 -mattr=+zilsd -M no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple riscv32 -mattr=+zilsd < %s \
# RUN:     | llvm-objdump  --mattr=+zilsd --no-print-imm-hex -M no-aliases -d -r - \
# RUN:     | FileCheck --check-prefix=CHECK-ASM-AND-OBJ %s

# CHECK-ASM-AND-OBJ: ld t1, 12(a0)
# CHECK-ASM: encoding: [0x03,0x33,0xc5,0x00]
ld t1, 12(a0)
# CHECK-ASM-AND-OBJ: ld a0, 4(a2)
# CHECK-ASM: encoding: [0x03,0x35,0x46,0x00]
ld a0, +4(a2)
# CHECK-ASM-AND-OBJ: ld t1, -2048(a4)
# CHECK-ASM: encoding: [0x03,0x33,0x07,0x80]
ld t1, -2048(a4)
# CHECK-ASM-AND-OBJ: ld t1, 2047(a4)
# CHECK-ASM: encoding: [0x03,0x33,0xf7,0x7f]
ld t1, 2047(a4)

# CHECK-ASM-AND-OBJ: sd s0, 2047(a0)
# CHECK-ASM: encoding: [0xa3,0x3f,0x85,0x7e]
sd s0, 2047(a0)
# CHECK-ASM-AND-OBJ: sd a0, -2048(a2)
# CHECK-ASM: encoding: [0x23,0x30,0xa6,0x80]
sd a0, -2048(a2)
