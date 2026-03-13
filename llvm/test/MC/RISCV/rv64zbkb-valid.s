# RUN: llvm-mc %s -triple=riscv64 -mattr=+zbkb -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+zbkb < %s \
# RUN:     | llvm-objdump --mattr=+zbkb --no-print-imm-hex -d -r - \
# RUN:     | FileCheck --check-prefix=CHECK-ASM-AND-OBJ %s

# CHECK-ASM-AND-OBJ: rev8 t0, t1
# CHECK-ASM: encoding: [0x93,0x52,0x83,0x6b]
rev8 t0, t1

# CHECK-ASM-AND-OBJ: rori t0, t1, 63
# CHECK-ASM: encoding: [0x93,0x52,0xf3,0x63]
rori t0, t1, 63
# CHECK-ASM-AND-OBJ: rorw t0, t1, t2
# CHECK-ASM: encoding: [0xbb,0x52,0x73,0x60]
rorw t0, t1, t2
# CHECK-ASM-AND-OBJ: rolw t0, t1, t2
# CHECK-ASM: encoding: [0xbb,0x12,0x73,0x60]
rolw t0, t1, t2
# CHECK-ASM-AND-OBJ: roriw t0, t1, 31
# CHECK-ASM: encoding: [0x9b,0x52,0xf3,0x61]
roriw t0, t1, 31
# CHECK-ASM-AND-OBJ: roriw t0, t1, 0
# CHECK-ASM: encoding: [0x9b,0x52,0x03,0x60]
roriw t0, t1, 0

# CHECK-ASM-AND-OBJ: packw t0, t1, t2
# CHECK-ASM: encoding: [0xbb,0x42,0x73,0x08]
packw t0, t1, t2

# Test the encoding used for zext.h on RV64
# CHECK-ASM-AND-OBJ: zext.h t0, t1
# CHECK-ASM: encoding: [0xbb,0x42,0x03,0x08]
packw t0, t1, zero

# Test the encoding used for zext.h on RV32
# CHECK-ASM-AND-OBJ: pack t0, t1, zero
# CHECK-ASM: encoding: [0xb3,0x42,0x03,0x08]
pack t0, t1, zero

# CHECK-ASM-AND-OBJ: zext.h t0, t1
# CHECK-ASM: encoding: [0xbb,0x42,0x03,0x08]
zext.h t0, t1
