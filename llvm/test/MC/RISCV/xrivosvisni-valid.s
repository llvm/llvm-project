# RUN: llvm-mc %s -triple=riscv32 -mattr=+experimental-xrivosvisni -riscv-no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv32 -mattr=+experimental-xrivosvisni < %s \
# RUN:     | llvm-objdump --mattr=+experimental-xrivosvisni -M no-aliases --no-print-imm-hex -d -r - \
# RUN:     | FileCheck --check-prefix=CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc %s -triple=riscv64 -mattr=+experimental-xrivosvisni -riscv-no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+experimental-xrivosvisni < %s \
# RUN:     | llvm-objdump --mattr=+experimental-xrivosvisni -M no-aliases --no-print-imm-hex -d -r - \
# RUN:     | FileCheck --check-prefix=CHECK-ASM-AND-OBJ %s

# CHECK-ASM-AND-OBJ: ri.vzero.v v1
# CHECK-ASM: encoding: [0xdb,0x70,0x00,0x00]
ri.vzero.v v1
# CHECK-ASM-AND-OBJ: ri.vzero.v v2
# CHECK-ASM: encoding: [0x5b,0x71,0x00,0x00]
ri.vzero.v v2
# CHECK-ASM-AND-OBJ: ri.vzero.v v3
# CHECK-ASM: encoding: [0xdb,0x71,0x00,0x00]
ri.vzero.v v3

# CHECK-ASM-AND-OBJ: ri.vinsert.v.x v0, zero, 0
# CHECK-ASM: encoding: [0x5b,0x60,0x00,0x40]
ri.vinsert.v.x v0, x0, 0
# CHECK-ASM-AND-OBJ: ri.vinsert.v.x	v1, s4, 13
# CHECK-ASM: encoding: [0xdb,0x60,0xda,0x40]
ri.vinsert.v.x v1, x20, 13
# CHECK-ASM-AND-OBJ: ri.vinsert.v.x	v1, zero, 1
# CHECK-ASM: encoding: [0xdb,0x60,0x10,0x40]
ri.vinsert.v.x v1, x0, 1
# CHECK-ASM-AND-OBJ: ri.vinsert.v.x	v23, ra, 1
# CHECK-ASM: encoding: [0xdb,0xeb,0x10,0x40]
ri.vinsert.v.x v23, x1, 1

# CHECK-ASM-AND-OBJ: ri.vextract.x.v	s4, v1, 13
# CHECK-ASM: encoding: [0x5b,0xaa,0x16,0x5e]
ri.vextract.x.v x20, v1, 13
# CHECK-ASM-AND-OBJ: ri.vextract.x.v	s5, v2, 31
# CHECK-ASM: encoding: [0xdb,0xaa,0x2f,0x5e]
ri.vextract.x.v x21, v2, 31
