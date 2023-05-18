# RUN: llvm-mc %s -triple=riscv32 -mattr=+m -riscv-no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc %s -triple=riscv64 -mattr=+m -riscv-no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv32 -mattr=+m < %s \
# RUN:     | llvm-objdump --mattr=+m -M no-aliases -d -r - \
# RUN:     | FileCheck --check-prefix=CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+m < %s \
# RUN:     | llvm-objdump --mattr=+m -M no-aliases -d -r - \
# RUN:     | FileCheck --check-prefix=CHECK-ASM-AND-OBJ %s

# CHECK-ASM-AND-OBJ: mul a4, ra, s0
# CHECK-ASM: encoding: [0x33,0x87,0x80,0x02]
mul a4, ra, s0
# CHECK-ASM-AND-OBJ: mulh ra, zero, zero
# CHECK-ASM: encoding: [0xb3,0x10,0x00,0x02]
mulh x1, x0, x0
# CHECK-ASM-AND-OBJ: mulhsu t0, t2, t1
# CHECK-ASM: encoding: [0xb3,0xa2,0x63,0x02]
mulhsu t0, t2, t1
# CHECK-ASM-AND-OBJ: mulhu a5, a4, a3
# CHECK-ASM: encoding: [0xb3,0x37,0xd7,0x02]
mulhu a5, a4, a3
# CHECK-ASM-AND-OBJ: div s0, s0, s0
# CHECK-ASM: encoding: [0x33,0x44,0x84,0x02]
div s0, s0, s0
# CHECK-ASM-AND-OBJ: divu gp, a0, a1
# CHECK-ASM: encoding: [0xb3,0x51,0xb5,0x02]
divu gp, a0, a1
# CHECK-ASM-AND-OBJ: rem s2, s2, s8
# CHECK-ASM: encoding: [0x33,0x69,0x89,0x03]
rem s2, s2, s8
# CHECK-ASM-AND-OBJ: remu s2, s2, s8
# CHECK-ASM: encoding: [0x33,0x79,0x89,0x03]
remu x18, x18, x24
